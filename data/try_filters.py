from collections import Counter, defaultdict
from datasets import load_dataset
from data.filters import keep, parser

ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/java",
                  split="train", streaming=True)

reasons, examples = Counter(), defaultdict(list)

for row in ds.take(1000):
    ok, why = keep(row)
    if ok and not parser(row["content"]):
        ok, why = False, "parse"
    reasons[why] += 1
    if not ok and len(examples[why]) < 2:
        examples[why].append(row["content"][:300])

print(reasons)
for why, ex in examples.items():
    for e in ex:
        print(f"--- dropped for: {why} ---\n{e}\n")
