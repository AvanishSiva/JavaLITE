import json, os
import pyarrow as pa, pyarrow.parquet as pq
from datasets import load_dataset
from kaggle_secrets import UserSecretsClient
from data.filters import keep, parses, split_of

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
TARGET_BYTES = int(float(os.environ.get("TARGET_BYTES", 8e9)))  
SHARD = 50_000                    
OUT = "/kaggle/working/clean"
os.makedirs(OUT, exist_ok=True)

ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/java", split="train",
                  streaming=True, token=os.environ["HF_TOKEN"])

funnel = {}
buf = {"train": [], "val": [], "test": []}
nshard = {"train": 0, "val": 0, "test": 0}
train_bytes = 0

def flush(split):
    if buf[split]:
        pq.write_table(pa.Table.from_pylist(buf[split]),
                       f"{OUT}/{split}-{nshard[split]:04d}.parquet")
        nshard[split] += 1
        buf[split] = []

for row in ds:
    funnel["seen"] = funnel.get("seen", 0) + 1
    ok, why = keep(row)
    if ok and not parses(row["content"]):
        ok, why = False, "parse"
    funnel[why] = funnel.get(why, 0) + 1
    if not ok:
        continue
    split = split_of(row["max_stars_repo_name"])
    buf[split].append({"content": row["content"],
                       "repo": row["max_stars_repo_name"],
                       "path": row["max_stars_repo_path"]})
    if split == "train":
        train_bytes += len(row["content"].encode("utf-8"))
    if len(buf[split]) >= SHARD:
        flush(split)
    if funnel["seen"] % 100_000 == 0:
        print(funnel, f"{train_bytes / 1e9:.2f} GB")
    if train_bytes >= TARGET_BYTES:
        break

for s in buf:
    flush(s)
json.dump(funnel, open(f"{OUT}/funnel.json", "w"), indent=2)
print("final funnel:", funnel)