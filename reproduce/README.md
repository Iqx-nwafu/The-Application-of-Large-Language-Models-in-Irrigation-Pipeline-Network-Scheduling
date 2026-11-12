# Re-judging with the paper's time-window tolerance

The per-decision records in `results/` store the judgement made while the experiments ran, when the time-window constraint
was checked at 1e-6 h. Group durations are written with 0.001 h resolution, so plans with 40 or more groups can exceed the
120 h window by 0.001-0.005 h through rounding alone. The paper judges the constraint with a 0.03 h tolerance for every
method. `python reproduce/rescore_counts.py` (simulation environment, `requirements/sim.txt`) applies that rule to the
stored violation records and writes:

| output | content | paper |
|---|---|---|
| `out/table4_counts.md` | feasibility before / after fallback for the workflow, NSGA-II and the current rotation table on the test and OOD sets, run-time -> tolerance | Table 4, Section 3.1 |
| `out/student_counts.md` | feasible before fallback, qualified, and final for every student evaluation; v3 vs v2 on val250; the teacher repro30 rows | Sections 3.4-3.5, Table 5 |

Rules (identical for all methods):

* A plan is re-judged feasible when its only violation is `T_window <= 0.03 h` and it has no transient, closing-time or
  checker error (`data.apply_twin_tol`; for the student records, the last pre-fallback check and the final check are
  re-judged the same way in `rescore_twindow_student.py`).
* Strict isolation-plan consistency: inconsistent fault plans are infeasible before fallback, and the safety layer (the
  NSGA-II + rule-isolation code) supplies the final plan. The tolerance is applied before this substitution.
* Time-outs are excluded; scenario O00237 (nothing left to schedule after isolation) is excluded.
* "Qualified" (student): feasible before fallback, consistent, deficit within tolerance of the NSGA-II reference and C* gap
  at most delta = 0.02.

Files: `data.py` (loads the round-1 records and applies the conventions above), `extract_checks.py` (condenses the student
records), `rescore_twindow_student.py` (student counts), `rescore_counts.py` (driver).
