import numpy as np
import glob
import random
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizer.fill_in_the_middle import fill_in_the_middle, unfill_in_the_middle
from tokenizer.clean import clean

tok = Tokenizer.from_file("tokenizer/javalite-16k/tokenizer.json")
EOT = tok.token_to_id("<|endoftext|>")
rng = random.Random(1337)

def encode_split(split, out_path, fim_rate = 0.5):
    n =0
    with open(out_path, "wb") as out:
        for path in sorted(glob.glob(f"/kaggle/input/datasets/sivaavanish/javalite-java-clean/clean/{split}-*.parquet")):
            code = pq.read_table(path, columns=["content"]).column("content").to_pylist()
            code  = [clean(c) for c in code]
            code = [fill_in_the_middle(c, rng) if rng.random() < fim_rate else c for c in code]
            for encoded in tok.encode_batch(code):
                ids = np.array(encoded.ids + [EOT], dtype=np.uint16)
                ids.tofile(out)
                n += len(ids)
            print(path, f"{n / 1e9:.3f}B tokens so far")

        return n

print("train tokens:", encode_split("train", "/kaggle/working/train.bin"))
print("val tokens:  ", encode_split("val", "/kaggle/working/val.bin"))