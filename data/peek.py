from datasets import load_dataset

ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/java",
                  split="train", streaming=True)
rows = list(ds.take(1000))

print(rows[0].keys())
sizes = sorted(r["size"] for r in rows)
print("median size:", sizes[len(sizes) // 2], "bytes | largest:", sizes[-1])

for r in rows[:3]:
    print("=" * 70)
    print(r["max_stars_repo_name"], "|", r["max_stars_repo_path"], "|", r["size"], "bytes")
    print(r["max_stars_repo_licenses"])
    print(r["content"][:600])