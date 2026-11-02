import pyarrow.parquet as pq, tiktoken
import glob
from tokenizers import Tokenizer
from transformers import AutoTokenizer

texts = pq.read_table(sorted(glob.glob("data/clean/test-*.parquet"))[0], columns=["content"]).column("content").to_pylist()[:500]

held = "\n\n".join(texts)
n_bytes = len(held.encode("utf-8"))

javalite_16k = Tokenizer.from_file("tokenizer/javalite-16k/tokenizer.json")
gpt2 = tiktoken.get_encoding("gpt2")
qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

rows =[
    ("gpt2", gpt2.n_vocab, len(gpt2.encode(held, disallowed_special=()))),
    ("qwen2.5-coder", len(qwen), len(qwen.encode(held))),
    ("javalite-16k", javalite_16k.get_vocab_size(), len(javalite_16k.encode(held).ids)),
]

print(f"{'tokenizer':14} {'vocab':>8} {'tokens':>10} {'bytes/token':>12}")
for name, v, n in rows:
    print(f"{name:14} {v:>8,} {n:>10,} {n_bytes / n:>12.2f}")