"""Re-judge the stored results with the paper's 0.03 h time-window tolerance and count feasible plans (read-only).

    python reproduce/rescore_counts.py        # -> reproduce/out/

The stored records carry the run-time judgement (time window checked at 1e-6 h). This script applies the tolerance used
in the paper to every method alike and prints the feasibility counts next to the run-time counts:
  * Table 4 (test and OOD): workflow before / after fallback, NSGA-II, current rotation table;
  * student evaluations (val250, scale and precision comparisons, local repro30 x 10 and precision runs), via
    extract_checks.py and rescore_twindow_student.py.
No figures and no statistical tests are produced."""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE))
import data as D          # noqa: E402

TASKS = ["switch", "rotation_plan", "fault_response", "rotation+fault"]
COLS = [("workflow, before fallback", "deepseek-v4-pro", "feasible_before"), ("workflow, final", "deepseek-v4-pro", "final_feasible"),
        ("NSGA-II, final", "nsga2", "final_feasible"), ("current rotation table, final", "rules", "final_feasible")]


def counts(df, set_name):
    sc = D.scenario_set(set_name)["all"]
    d = D.main_rows(df)
    d = d[d.scenario.isin(sc)]
    out = {}
    for t in TASKS:
        x = d[d.task.isin(["rotation_plan", "fault_response"] if t == "rotation+fault" else [t])]
        for lab, e, col in COLS:
            v = x[x.entity == e][col].dropna().astype(bool)
            out[(t, lab)] = (int(v.sum()), len(v))
    return out


def api():
    tabs = {}
    for tol in (None, 0.03):
        D.CONFIG.update(iso="fail", timeout="exclude", twin_tol=tol)
        df = D.load()
        tabs[tol] = {s: counts(df, s) for s in ("test", "ood150")}
    L = ["# Table 4: feasibility counts, run-time judgement -> 0.03 h time-window tolerance\n",
         "Strict isolation-plan consistency; time-outs excluded; scenario O00237 excluded.\n"]
    for s, name in (("test", "Test set"), ("ood150", "OOD set")):
        L += [f"\n## {name}\n", "| task | " + " | ".join(c[0] for c in COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
        for t in TASKS:
            cells = []
            for lab, _, _ in COLS:
                (a, n), (b, _) = tabs[None][s][(t, lab)], tabs[0.03][s][(t, lab)]
                cells.append(f"{a}/{n}" if a == b else f"{a}/{n} -> {b}/{n}")
            L.append(f"| {t} | " + " | ".join(cells) + " |")
    (OUT / "table4_counts.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def run(*args):
    print("$", " ".join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


def student():
    st = ROOT / "results" / "student"
    tags = sorted(p.name[:-len(".jsonl.gz")] for p in st.glob("*.jsonl.gz"))
    local = [t for t in tags if not t.endswith("_cloud")]
    cloud = [t for t in tags if t.endswith("_cloud")]
    run(HERE / "extract_checks.py", OUT / "checks_local.jsonl", *[st / f"{t}.jsonl.gz" for t in local])
    run(HERE / "extract_checks.py", OUT / "checks_cloud.jsonl", *[st / f"{t}.jsonl.gz" for t in cloud])
    run(HERE / "rescore_twindow_student.py", ROOT / "src", OUT / "student_counts.md", OUT / "checks_local.jsonl", OUT / "checks_cloud.jsonl")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    api()
    student()
