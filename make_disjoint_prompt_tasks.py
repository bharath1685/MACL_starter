#!/usr/bin/env python3
"""Create non-overlapping prompt-level Task A/B datasets without changing originals."""
import json
from pathlib import Path

ROOT = Path.cwd()
SRC = ROOT / "data" / "continual_tasks"
OUT = ROOT / "data" / "continual_tasks_disjoint_prompt"

if not SRC.is_dir():
    raise SystemExit(f"Missing input directory: {SRC}")
if OUT.exists():
    raise SystemExit(f"Refusing to overwrite existing directory: {OUT}")

names = ("task_a_train.jsonl", "task_a_val.jsonl",
         "task_b_train.jsonl", "task_b_val.jsonl")

def read_rows(name):
    path = SRC / name
    if not path.is_file():
        raise SystemExit(f"Missing required file: {path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row.get("prompt"), str) or not isinstance(row.get("answer"), str):
                    raise SystemExit(f"{path}:{line_no}: expected string prompt and answer")
                rows.append(row)
    return rows

data = {name: read_rows(name) for name in names}
a_prompts = {r["prompt"] for n in names[:2] for r in data[n]}
b_prompts = {r["prompt"] for n in names[2:] for r in data[n]}
overlap = a_prompts & b_prompts

# Remove any exact prompt appearing in both tasks from BOTH tasks.
# Also deduplicate exact prompts within each file, preserving first occurrence.
clean = {}
for name in names:
    is_a = name.startswith("task_a")
    rows = []
    seen = set()
    for row in data[name]:
        p = row["prompt"]
        if p in overlap or p in seen:
            continue
        seen.add(p)
        rows.append(row)
    clean[name] = rows

# Ensure train/validation prompt separation within each task as a safeguard.
for prefix in ("task_a", "task_b"):
    train_name, val_name = f"{prefix}_train.jsonl", f"{prefix}_val.jsonl"
    train_prompts = {r["prompt"] for r in clean[train_name]}
    clean[val_name] = [r for r in clean[val_name] if r["prompt"] not in train_prompts]

OUT.mkdir(parents=True)
for name, rows in clean.items():
    with (OUT / name).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

report = {
    "method": "remove exact prompt overlap across Task A/B; deduplicate prompts within files; keep task train/validation prompts separate",
    "input_dir": str(SRC),
    "output_dir": str(OUT),
    "original_counts": {n: len(data[n]) for n in names},
    "output_counts": {n: len(clean[n]) for n in names},
    "cross_task_prompt_overlap_removed": len(overlap),
    "remaining_cross_task_prompt_overlap": len(
        ({r["prompt"] for n in names[:2] for r in clean[n]}) &
        ({r["prompt"] for n in names[2:] for r in clean[n]})
    ),
    "warning": "Prompt-level disjointness does not guarantee scenario-level disjointness or distinct reward-optimal behavior. This is a diagnostic improvement, not proof that the tasks conflict."
}
with (OUT / "overlap_audit.json").open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
