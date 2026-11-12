"""级联第二级清单：to_sft 判为不可用的 Flash 轨迹（只取 status=ok 的；超时 / 报错用原教师 --retry-failed 重跑，退化不跑），且 raw/ 里还没有 Pro 轨迹 → 一行一个 job_id，交 distill_gen --ids。

python distill/to_sft.py --src distill/raw_flash --out data_flash --max-tokens 24576 --repair      # 先补齐 GA 参照（--ref-shard）
python distill/redo_list.py distill/data_flash/index.jsonl > distill/plans/redo_<日期>.txt
python distill/distill_gen.py --plan plan_10000 --ids distill/plans/redo_<日期>.txt --budget-cny ...   # Pro 写 raw/（另有应力切换时 --plan plan_stress 再跑一遍）
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
for l in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(l)
    if not r["keep"] and r["status"] == "ok" and not (HERE / "raw" / r["task"] / f"{r['job_id']}.json").exists():
        print(r["job_id"])
