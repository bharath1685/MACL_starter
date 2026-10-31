from pathlib import Path

DATA = (
    Path.home()
    / "Projects"
    / "MACL_starter"
    / "data_source"
    / "end-to-end-negotiator"
    / "src"
    / "data"
    / "negotiate"
)

print("Dataset directory:", DATA)
print("Directory exists:", DATA.exists())

for filename in ["train.txt", "val.txt", "test.txt", "data.txt"]:
    path = DATA / filename

    if not path.exists():
        print(f"\n{filename}: not found")
        continue

    print(f"\n--- {filename} ---")
    print("Size:", round(path.stat().st_size / 1024, 2), "KB")
    print("Line count:", sum(1 for _ in path.open(encoding="utf-8")))

    with path.open(encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i == 5:
                break
            print(f"Line {i + 1}: {line.rstrip()[:250]}")
