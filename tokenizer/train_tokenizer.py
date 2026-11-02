import glob, os
import pyarrow.parquet as pq
from tokenizers import ByteLevelBPETokenizer

input_dir = "/kaggle/input/datasets/sivaavanish/javalite-java-clean/clean"

def clean(code):
    return "\n".join(line.rstrip() for line in code.replace("\r\n", "\n").split("\n"))

SPECIAL = [
    "<|endoftext|>",                                       # 0: end of a file
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",  # 1-3: fill-in-the-middle
    "<|pad|>",                                             # 4: padding
    "<|instruction|>", "<|code|>", "<|edit|>",             # 5-7: edit mode 
    "<|search|>", "<|replace|>", "<|end_edit|>",           # 8-10: edit blocks
]

#Take first three files and store it on tok_sample.txt

def java_files(max_bytes=1_000_000_000):
    seen = 0
    for path in sorted(glob.glob(f"{input_dir}/train-*.parquet")):
        for code in pq.read_table(path, columns=["content"]).column("content").to_pylist():
            code = clean(code)
            yield code
            seen += len(code)
            if seen >= max_bytes:
                return



with open("tok_sample.txt", "w", encoding="utf-8") as f:
    for path in sorted(glob.glob(f"{input_dir}/train-*.parquet"))[:3]:
        for code in pq.read_table(path, columns = ["content"]).column("content").to_pylist():
            f.write(code + "\n")

tok = ByteLevelBPETokenizer()
tok.train_from_iterator(java_files(), vocab_size=16384, min_frequency=2, special_tokens=SPECIAL)

os.makedirs("tokenizer/javalite-16k", exist_ok=True)
tok.save_model("tokenizer/javalite-16k")            
tok.save("tokenizer/javalite-16k/tokenizer.json") 
print("vocab:", tok.get_vocab_size(), "| <|endoftext|> =", tok.token_to_id("<|endoftext|>"))

sample = "class A {\n    void f() {\n        return;\n    }\n}\n"
print([tok.decode([i]) for i in tok.encode(sample).ids]) 