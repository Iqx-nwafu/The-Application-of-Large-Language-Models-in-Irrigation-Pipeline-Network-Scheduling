"""SFT 样本的 Qwen3 渲染与损失掩码（train_sft.py 与 build_sft.py 共用，保证统计的长度就是训练时的长度）。"""
from transformers import AutoTokenizer

KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")


class Encoder:
    def __init__(self, model):
        self.tok = AutoTokenizer.from_pretrained(model)
        self.im_s, self.im_e = self.tok.convert_tokens_to_ids("<|im_start|>"), self.tok.convert_tokens_to_ids("<|im_end|>")
        self.a_hdr = self.tok.encode("assistant\n", add_special_tokens=False)

    def __call__(self, r):
        """整段渲染一次（非思考模式），按 <|im_start|>assistant\\n … <|im_end|> 找 assistant 段，依次对应 messages 里的 assistant 轮
        （Qwen3 模板在新 user 轮出现时会改写历史轮的空 think 块，逐段渲染不前缀一致）。返回 token 与"该位置计损失"掩码；
        train=false 的 assistant 轮（出错轮，只作上下文）不计损失。"""
        msgs = [{k: v for k, v in m.items() if k in KEYS} for m in r["messages"]]
        ids = self.tok.apply_chat_template(msgs, tools=r.get("tools"), tokenize=True, enable_thinking=False)
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids          # transformers 5 返回 BatchEncoding
        flags = [m.get("train", True) for m in r["messages"] if m["role"] == "assistant"]
        mask, k, i = [False] * len(ids), 0, 0
        while i < len(ids):
            if ids[i] == self.im_s and ids[i + 1:i + 1 + len(self.a_hdr)] == self.a_hdr:
                j, s = ids.index(self.im_e, i), i + 1 + len(self.a_hdr)
                if flags[k]:
                    mask[s:j + 1] = [True] * (j + 1 - s)
                k, i = k + 1, j
            i += 1
        assert k == len(flags), (k, len(flags))
        return ids, mask
