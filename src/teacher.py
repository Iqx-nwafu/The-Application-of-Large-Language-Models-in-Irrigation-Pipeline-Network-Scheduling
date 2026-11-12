"""教师模型数据生成接口：约束增强提示（沿用 Chen 等 2026 的六段式）+ 工具调用循环 + 校核反馈 + 兜底 + 轨迹日志。

决策模型通过 OpenAI 兼容接口调用（DeepSeek、Kimi/Moonshot、通义千问/DashScope 官方都是此格式，--provider 选）。
密钥只从环境变量读取（DEEPSEEK_API_KEY / MOONSHOT_API_KEY / DASHSCOPE_API_KEY，可用 --key-env 改名），不写入任何文件或日志。

一个场景的流程
  1. 由场景构造状态表示（管网摘要 + 工况 + 任务），拼成六段提示（--ablate 去掉第 3、4 段做消融）
  2. 模型可多次调用工具（simulate_steady / screen_transient / simulate_transient / check_constraints / query_node / run_optimizer）
  3. 模型给出最终 JSON → check_constraints 全量校核；不通过则把结构化反馈发回，最多 K 轮
  4. K 轮后仍不可行 → 安全兜底层接管（fallback_used 单独统计）
  5. 整条轨迹（消息、工具调用与返回、token、时延、校核结果）写入 JSONL

python teacher.py --task switch --n 3 --dry-run            # 不调 API，用规则桩跑通流程
python teacher.py --task rotation_plan --n 20 --split train        # 默认 deepseek-v4-pro
"""
import argparse
import json
import os
import pathlib
import time
import numpy as np
import jsonschema
import network as net
import evaluate as ev
import transient_tools as tt
import tools

HERE = pathlib.Path(__file__).parent
SCHEMA = json.load(open(HERE / "decision_output.schema.json", encoding="utf-8"))
TEMPLATES = json.load(open(HERE / "valve_templates.json", encoding="utf-8"))
K_ROUNDS = 3                      # 修正轮数（作者 2026-10-27 定）
MAX_TOOL_CALLS = 12               # 每轮工具调用上限
WINDOW_S = {"rotation_plan": 600.0, "switch": 120.0, "fault_response": 300.0}   # 决策窗口；超窗只计"实时未达标"，不中断。分组 600 s 只作报告不进 r0（作者 2026-10-28）
REQUEST_TIMEOUT_S = 1200          # 单次请求硬超时 20 min；超时只补发一次（作者 2026-10-28 要求防止无限等待）
BASE_URLS = {"deepseek": "https://api.deepseek.com", "moonshot": "https://api.moonshot.cn/v1",
             "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",          # 通义千问（阿里云百炼）
             "openai": "https://api.openai.com/v1"}                                     # 国际对照，不作教师
KEY_ENVS = {"deepseek": "DEEPSEEK_API_KEY", "moonshot": "MOONSHOT_API_KEY", "dashscope": "DASHSCOPE_API_KEY",
            "openai": "OPENAI_API_KEY"}


# ---------------- 提示 ----------------
def network_summary(h_source=None):
    """h_source：难瞬态子集的高水头场景给出实际水源水头；None 时为 net.H_SOURCE（主实验原文不变）。"""
    sub = "、".join(f"{L:.0f}" for L in net.SUB_LEN)
    return (
        f"树状有压滴灌管网（呼图壁联丰村二队 2 号系统，71 个工程节点，860 亩）。水源 Hub 定水头 {h_source or net.H_SOURCE:.0f} m。\n"
        f"干管 Hub–J0–J1–…–J9（PVC-U Φ250，内径 235.4 mm，段长 50 + 9×100 m，高程 0，PN 0.6 MPa ≈ 61.2 m）。\n"
        f"每个干管节点 Jx 接一条分干管：首端阀 VSx（DN150，最短启闭 10 s）→ Jx1…Jx6（PVC-U Φ160，内径 150.4 mm，"
        f"段长 {sub} m，高程沿程降到 −3.5 m）。\n"
        f"每个分干管节点 Jxk 两侧各一条支管：支管阀 Vxk{{L,R}}（DN80，最短启闭 5 s）→ PE Φ90 支管 60 m（PN 0.25 MPa ≈ 25.5 m）"
        f"→ 末端 Exk{{L,R}} 滴灌出流，q = 47·(p/10)^0.5 m³/h。共 120 个支管阀，编号 V{{x}}{{k}}{{L|R}}，x=0..9，k=1..6。\n"
        f"观测：干管 J0–J9 压力 10 个 + 干管各段流量 + 首部流量。"
    )


def hard_constraints():
    return (
        f"1. 稳态：开启支管末端压力 ≥ {ev.H_MIN:.0f} m；首部流量 ≤ {ev.Q_MAX:.0f} m³/h；干管、分干管流速 ≤ {ev.V_MAX} m/s；"
        f"PE 支管（阀后入口 B 和末端 E）稳态压力 ≤ {ev.PN_PE - ev.PE_MARGIN} m（PN {ev.PN_PE} m 留 {ev.PE_MARGIN:.0f} m 瞬态裕量），PVC 管 ≤ {ev.PN_PVC} m。\n"
        f"2. 瞬态（每次切换、每次故障处置）：阀前 PVC 区最大压力 ≤ {tt.ALPHA}×61.2 m，阀后 PE 支管 ≤ {tt.ALPHA}×25.5 m；"
        f"阀前 PVC 区最小压力 ≥ {tt.H_VAC} m。\n"
        f"3. 启闭历时：支管阀 ≥ 5 s，分干管阀 ≥ 10 s；常规切换不得使用 instant。\n"
        f"4. 每个可用支管阀恰好属于一个轮灌组，每组 1–8 个阀；被隔离或卡关的支管阀不得入组。\n"
        f"5. 轮灌总时长（含每次切换 2 min）≤ {ev.T_WIN:.0f} h；每组灌水时长 0.5–8 h。\n"
        f"6. 灌水量：支管 i 的需水 D_i（m³）应由实际流量 × 组灌水时长满足。\n"
        f"7. 支管阀开度不由你决定：稳态计算会把入口压力超过 {ev.P_B_CAP} m 的支管阀自动节流到该值"
        f"（simulate_steady 返回各组 openings），开阀模板开到该目标开度即可。\n"
        f"目标：缺水量最小优先；缺水相同时比较灌水均匀度 CV_q、轮灌总时长、超灌水量，用 TOPSIS 贴近度 C*（simulate_steady 返回 score，越大越好）。"
    )


def transient_rules():
    return "\n".join(f"{i + 1}. {h}" for i, h in enumerate(TEMPLATES["heuristics"])) + \
        "\n模板参数只能取模板库的离散档：\n" + json.dumps(
            {t["id"]: t.get("allowed", {}).get("branch") for t in TEMPLATES["templates"] if t.get("allowed")},
            ensure_ascii=False)


def task_text(task, sc, state):
    D = sc["demand_m3"]
    dem = "；".join(f"{v}:{D[v]:.0f}" for v in ev.ALL_VALVES)
    faults = sc.get("faults", [])
    base = f"场景 {sc['id']}：第 {sc['irrigation_event']} 次灌水，定额 {sc['quota_m3_per_mu']} m³/亩。各支管需水 D_i（m³）：{dem}。"
    if task == "rotation_plan":
        return base + "\n任务：给出全部 120 条支管的轮灌分组、组序和每组灌水时长（decision_type = rotation_plan）。"
    if task == "switch":
        return base + (f"\n当前运行组 {state['from_group']}：{state['g1']}；下一组 {state['to_group']}：{state['g2']}。"
                       f"\n任务：给出这次切换中每个阀门的动作（起始时刻、模板、参数），decision_type = switch。")
    f = faults[0]
    desc = {"burst": f"管段 {f.get('pipe')} 爆管（节点 {f.get('node')} 附近）",
            "leak": f"管段 {f.get('pipe')} 泄漏约 {f.get('q_ref_m3h')} m³/h",
            "stuck": f"阀门 {f.get('valve')} 卡死（{f.get('mode')}{'，开度 ' + str(f.get('opening')) if f.get('opening') else ''}）"}[f["type"]]
    return base + (f"\n故障检测模块报告：{desc}，置信度 0.9。故障发生时正在运行的组：{state['g1']}。"
                   f"已灌完的支管：{state['done']}。剩余时间窗 {state['t_left_h']:.1f} h。"
                   f"\n任务：给出隔离阀与隔离动作、退出灌溉的支管、剩余支管的轮灌计划（decision_type = fault_response）。")


def build_prompt(task, sc, state, ablate=False, fewshot=None, with_tools=True):
    secs = [("角色与任务", "你是灌区调度工程师，负责有压滴灌管网的轮灌分组、组间切换的阀门启闭和故障处置。"
                        + ("可以调用工具做稳态、瞬态计算和约束检查。" if with_tools else "没有计算工具，只能凭管网信息和约束直接给出方案。")
                        + "先推理再决策，最终只输出一个 JSON。"),
            ("管网摘要", network_summary(sc.get("perturbation", {}).get("h_source")))]
    if not ablate:
        secs += [("硬约束", hard_constraints()), ("瞬态经验规则", transient_rules())]
    secs += [("输出格式", "最终回答只包含一个符合下列 JSON Schema 的对象，不要输出其他文字：\n" + json.dumps(SCHEMA, ensure_ascii=False))]
    if fewshot:
        secs += [("示例", json.dumps(fewshot, ensure_ascii=False))]
    system = "\n\n".join(f"【{i + 1}. {t}】\n{c}" for i, (t, c) in enumerate(secs))
    return system, task_text(task, sc, state)


# ---------------- 场景 → 决策状态 ----------------
def make_state(task, sc, rng):
    ctx = ev.Context(sc, apply_faults=(task == "fault_response"))
    st = dict()
    base = net.rotation_groups()
    if task == "switch" and sc.get("switch_pair"):     # 难瞬态子集：指定切换的两组（预设轮灌表组号，0 起）
        k1, k2 = sc["switch_pair"]
        st.update(from_group=k1 + 1, to_group=k2 + 1, g1=base[k1], g2=base[k2])
    elif task == "switch":
        k = int(rng.integers(len(base) - 1))
        st.update(from_group=k + 1, to_group=k + 2, g1=base[k], g2=base[k + 1])
    elif task == "fault_response":
        f = sc["faults"][0]
        frac = f.get("cycle_frac", f.get("switch_frac", 0.0))
        k = min(int(frac * len(base)), len(base) - 1)
        st.update(g1=base[k], done=[v for g in base[:k] for v in g], t_left_h=ev.T_WIN * (1 - frac))
        ctx.t_win = st["t_left_h"]
        ctx.valves = [v for v in ctx.valves if v not in st["done"]]
    return ctx, st


# ---------------- 模型调用 ----------------
class EmptyResponse(Exception):
    """服务端过载时偶尔返回空响应（2026-11-03 DeepSeek 空闲时段开头，r 为 None 或无 choices），按可重试错误处理。"""


class Client:
    def __init__(self, model, provider="deepseek", key_env=None, dry_run=False):
        self.model, self.dry, self.provider = model, dry_run, provider
        self.temperature = None if provider in ("moonshot", "openai") else 0.6   # Kimi K3 / GPT 推理模型不接受自定义温度
        key_env = key_env or KEY_ENVS[provider]
        if not dry_run:
            from openai import OpenAI
            key = os.environ.get(key_env)
            if not key:
                raise SystemExit(f"环境变量 {key_env} 未设置")
            # 千问 Max 在分组长提示上单次回复可超 10 min（默认超时）；硬超时 REQUEST_TIMEOUT_S，客户端自带重试关掉，统一走 _retry
            self.cli = OpenAI(api_key=key, base_url=BASE_URLS[provider], timeout=REQUEST_TIMEOUT_S, max_retries=0)

    def _retry(self, fn):
        """429 / 过载 / 5xx / 连接错误 / 空响应：指数退避 30、60、120、240、240 s，最多重试 5 次（2026-11-05 由 3 次放宽：
        DeepSeek 空闲时段开头会成串 503 约半小时）；超时（REQUEST_TIMEOUT_S）只补一次，再超时就抛 APITimeoutError 由 evalrun 记 status=timeout。
        原始错误体追加到 results/api_errors.log（不含密钥）；等待时长记进 usage.retry_wait_s，时延统计可扣除。"""
        import openai
        waited, n_timeout = 0.0, 0
        for i in range(6):
            try:
                return fn(), waited
            except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError, openai.APITimeoutError, EmptyResponse) as e:
                timeout = isinstance(e, openai.APITimeoutError)
                n_timeout += timeout
                body = getattr(e, "body", None) or str(e)
                with open(HERE / "results" / "api_errors.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), provider=self.provider, model=self.model,
                                            attempt=i + 1, error=type(e).__name__, body=body), ensure_ascii=False, default=str) + "\n")
                if i == 5 or n_timeout > 1:
                    raise
                w = 0 if timeout else 30 * 2 ** min(i, 3)
                time.sleep(w); waited += w

    def chat(self, messages, tools_spec):
        """统一返回 (chat 格式的 assistant 消息, usage, 时延)。OpenAI 推理模型的函数调用只走 Responses 接口。"""
        t0 = time.time()
        if self.dry:
            return _stub_reply(messages), dict(prompt_tokens=0, completion_tokens=0), time.time() - t0
        if self.provider == "openai":
            (reply, u), waited = self._retry(lambda: self._responses(messages, tools_spec))
            u["retry_wait_s"] = waited
            return reply, u, time.time() - t0
        kw = dict(tools=[{"type": "function", "function": s} for s in tools_spec]) if tools_spec else {}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        def call():
            r = self.cli.chat.completions.create(model=self.model, messages=messages, **kw)
            if r is None or not getattr(r, "choices", None):
                raise EmptyResponse("API 返回空响应")
            return r
        r, waited = self._retry(call)
        u = r.usage.model_dump() if r.usage else {}
        u["served_model"] = r.model                                   # 记录实际快照号
        u["retry_wait_s"] = waited
        return r.choices[0].message.model_dump(exclude_none=True), u, time.time() - t0

    def _responses(self, messages, tools_spec):
        """把 chat 格式的对话转成 Responses 接口的 input（无状态：推理项带 encrypted_content 原样回传）。"""
        instructions = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
        items = []
        for m in messages[1:]:
            if m["role"] == "user":
                items.append(dict(role="user", content=m["content"]))
            elif m["role"] == "assistant":
                items.extend(m.get("_raw_items") or ([dict(role="assistant", content=m["content"])] if m.get("content") else []))
            elif m["role"] == "tool":
                items.append(dict(type="function_call_output", call_id=m["tool_call_id"], output=m["content"]))
        kw = dict(tools=[dict(type="function", name=t["name"], description=t["description"], parameters=t["parameters"])
                         for t in tools_spec]) if tools_spec else {}
        r = self.cli.responses.create(model=self.model, instructions=instructions, input=items, store=False,
                                      include=["reasoning.encrypted_content"], **kw)
        raw, calls, text = [], [], []
        for it in r.output:
            d = it.model_dump(exclude_none=True)
            raw.append(d)
            if it.type == "function_call":
                calls.append(dict(id=it.call_id, type="function", function=dict(name=it.name, arguments=it.arguments)))
            elif it.type == "message":
                text.append("".join(getattr(c, "text", "") for c in it.content))
        reply = dict(role="assistant", content="\n".join(t for t in text if t) or None, _raw_items=raw)
        if calls:
            reply["tool_calls"] = calls
        u = r.usage.model_dump() if r.usage else {}
        usage = dict(prompt_tokens=u.get("input_tokens"), completion_tokens=u.get("output_tokens"),
                     prompt_tokens_details=dict(cached_tokens=(u.get("input_tokens_details") or {}).get("cached_tokens")),
                     completion_tokens_details=dict(reasoning_tokens=(u.get("output_tokens_details") or {}).get("reasoning_tokens")),
                     served_model=r.model)
        return reply, usage


def _stub_reply(messages):
    """离线桩：读出任务里的组，用工程规则直接给答案（只用于跑通流程）。"""
    task = messages[1]["content"]
    if "decision_type = switch" in task:
        g1 = json.loads(task.split("当前运行组")[1].split("：", 1)[1].split("；")[0].replace("'", '"'))
        g2 = json.loads(task.split("下一组")[1].split("：", 1)[1].split("。")[0].replace("'", '"'))
        d = dict(schema_version="0.1", scenario_id="stub", decision_type="switch", reasoning="规则桩：先开后关，两阶段关阀",
                 switch=dict(from_group=1, to_group=2, actions=tools.rule_switch(g1, g2, 0)))
        return dict(role="assistant", content=json.dumps(d, ensure_ascii=False))
    return dict(role="assistant", content="{}")


def execute_tool(name, args, ctx, sc, state):
    if name == "simulate_steady":
        return tools.simulate_steady(ctx, args["rotation_plan"])
    if name == "screen_transient":
        return tools.screen_transient(ctx, args["open_before"], args["actions"])
    if name == "simulate_transient":
        return tools.simulate_transient(ctx, args["open_before"], args["actions"])
    if name == "check_constraints":
        return tools.check_constraints(ctx, args["decision"], args.get("open_before", state.get("g1")))
    if name == "query_node":
        return tools.query_node(args["name"])
    if name == "run_optimizer":
        return tools.run_optimizer(ctx, sc, apply_faults=bool(sc.get("faults")))
    return dict(error=f"未知工具 {name}")


def parse_decision(text):
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`").split("\n", 1)[1].rsplit("```", 1)[0] if "\n" in s else s
    i, j = s.find("{"), s.rfind("}")
    d = json.loads(s[i:j + 1])
    jsonschema.validate(d, SCHEMA)
    return d


def solve(client, task, sc, rng, ablate=False, k_rounds=None, tool_specs=None, skip_transient=False, max_tool_calls=None):
    """ablate: 去掉硬约束与经验规则两段；k_rounds: 修正轮数（0 = 直出）；tool_specs: 可用工具（[] = 无工具）；
    skip_transient: 校核反馈里不做瞬态（消融）；max_tool_calls: 每轮工具上限（缺省 = 主实验口径 MAX_TOOL_CALLS）。"""
    k_rounds = K_ROUNDS if k_rounds is None else k_rounds
    cap = MAX_TOOL_CALLS if max_tool_calls is None else max_tool_calls
    tool_specs = tools.TOOL_SPECS if tool_specs is None else tool_specs
    ctx, state = make_state(task, sc, rng)
    system, user = build_prompt(task, sc, state, ablate, with_tools=bool(tool_specs))
    msgs = [dict(role="system", content=system), dict(role="user", content=user)]
    log = dict(scenario=sc["id"], split=sc.get("split"), task=task, model=client.model, ablate=ablate,
               k_rounds=k_rounds, n_tools=len(tool_specs), skip_transient=skip_transient, max_tool_calls=cap,
               state={k: v for k, v in state.items()}, rounds=[], tool_calls=0, usage=[], latency_s=0.0)
    t_start = time.time()
    first_feasible_s, decision, ok = None, None, False
    for rnd in range(k_rounds + 1):
        n_calls = 0
        while True:
            reply, usage, dt = client.chat(msgs, tool_specs)
            log["usage"].append(usage); log["latency_s"] += dt
            msgs.append(reply)
            calls = reply.get("tool_calls") or []
            if not calls:
                break
            over = n_calls + len(calls) > cap
            for c in calls:                                           # 每个 tool_call 都必须有回应，否则接口报 400
                n_calls += 1; log["tool_calls"] += 1
                if over:
                    res = dict(error=f"工具调用已达上限 {cap} 次，请直接给出最终 JSON")
                else:
                    try:
                        res = execute_tool(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"), ctx, sc, state)
                    except Exception as e:                            # 工具报错也作为反馈交给模型
                        res = dict(error=f"{type(e).__name__}: {e}")
                msgs.append(dict(role="tool", tool_call_id=c["id"], content=json.dumps(res, ensure_ascii=False, default=float)))
            if over:
                break
        submitted = None
        try:
            decision = submitted = parse_decision(reply.get("content") or "")
            chk = tools.check_constraints(ctx, decision, open_before=state.get("g1"), skip_transient=skip_transient)
        except Exception as e:
            chk = dict(feasible=False, format_error=f"{type(e).__name__}: {str(e)[:300]}")
        log["rounds"].append(dict(round=rnd, check=chk, decision=submitted))     # decision：本轮提交的方案（解析失败为 None），评测事后做安全口径
        if chk["feasible"]:
            ok = True
            first_feasible_s = time.time() - t_start
            break
        if rnd < k_rounds:
            msgs.append(dict(role="user", content="校核未通过，请根据以下结构化反馈修改方案，仍只输出 JSON：\n" +
                             json.dumps(chk, ensure_ascii=False, default=float)))
    log.update(feasible_before_fallback=ok, n_rounds=len(log["rounds"]), first_feasible_s=first_feasible_s,
               model_wall_s=time.time() - t_start, decision=decision, messages=msgs)
    if not ok:                                                        # 安全兜底层
        if task == "switch":
            acts, r, lv = tools.safe_switch(ctx, state["g1"], state["g2"])
            log.update(fallback_used="safe_switch", fallback_level=lv, fallback_feasible=r["feasible"],
                       final_decision=dict(decision_type="switch", switch=dict(actions=acts)))
        else:
            plan, src = tools.fallback_rotation(ctx, sc)
            if task == "rotation_plan":
                fd = dict(decision_type="rotation_plan", rotation_plan=plan)
            elif ctx.iso_valves:                                       # 难瞬态子集：隔离关阀也要过瞬态，逐档放慢
                fd, lv = tools.safe_fault(ctx, sc, plan, open_before=state.get("g1") or ())
                log["fallback_level"] = lv
            else:
                fd = tools.rule_fault(ctx, sc, plan)
            log.update(fallback_used=src, final_decision=fd)
    else:
        log.update(fallback_used=None, final_decision=decision)
    wall = time.time() - t_start                                       # 端到端时延（含兜底）
    log.update(wall_s=wall, window_s=WINDOW_S[task], in_window=wall <= WINDOW_S[task])
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rotation_plan", "switch", "fault_response"], default="switch")
    ap.add_argument("--model", default="deepseek-v4-pro")   # 作者 2026-10-27：各家用顶级型号
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--key-env", default=None, help="默认按 provider 取 DEEPSEEK_API_KEY / MOONSHOT_API_KEY")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    kind = "fault" if a.task == "fault_response" else "nominal"
    S = [json.loads(l) for l in open(HERE / "scenarios.jsonl", encoding="utf-8")]
    S = [s for s in S if s["split"] == a.split and s["kind"] == kind][:a.n]
    client = Client(a.model, a.provider, a.key_env, a.dry_run)
    rng = np.random.default_rng(a.seed)
    out = HERE / "trajectories" / f"{a.task}_{'dry' if a.dry_run else a.model}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for sc in S:
            log = solve(client, a.task, sc, rng, a.ablate)
            f.write(json.dumps(log, ensure_ascii=False, default=float) + "\n")
            tok = sum((u.get("prompt_tokens", 0) or 0) + (u.get("completion_tokens", 0) or 0) for u in log["usage"])
            print(f"{sc['id']}: 兜底前可行 {log['feasible_before_fallback']}，轮数 {log['n_rounds']}，工具调用 {log['tool_calls']}，"
                  f"token {tok}，模型时延 {log['latency_s']:.1f} s，总时长 {log['wall_s']:.1f} s（窗口 {log['window_s']:.0f} s，"
                  f"{'达标' if log['in_window'] else '未达标'}），兜底 {log['fallback_used']}")
    print("→", out.relative_to(HERE))


if __name__ == "__main__":
    main()
