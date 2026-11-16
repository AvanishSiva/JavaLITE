
import os, re, json, glob, shutil, subprocess, tempfile, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"
JAVATUPLES = os.path.abspath("javatuples-1.2.jar")
SHARDS_DIR = "shards"
GEN_FILE   = "qwen_completions.json"
OUT_FILE   = "qwen_humaneval_java.json"
LIMIT      = int(sys.argv[1]) if len(sys.argv) > 1 else None
MAX_NEW, RUN_TIMEOUT, COMPILE_TIMEOUT, WORKERS = 512, 10, 60, 4


ds = load_dataset("nuprl/MultiPL-E", "humaneval-java", split="test")
if LIMIT: ds = ds.select(range(LIMIT))

 
cache = json.load(open(GEN_FILE)) if os.path.exists(GEN_FILE) else {}
todo = [p for p in ds if p["name"] not in cache]
if todo:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if dev == "cuda" else torch.float32
    ).to(dev).eval()

    def generate(prompt, stops):
        ids = tok(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                stop_strings=stops,
                tokenizer=tok,
                pad_token_id=tok.eos_token_id,
            )

        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        cuts = [text.find(s) for s in stops if s in text]
        return text[:min(cuts)] if cuts else text

    for i, p in enumerate(todo):
        cache[p["name"]] = generate(p["prompt"], p["stop_tokens"])
        json.dump(cache, open(GEN_FILE, "w"))         
        print(f"gen {i+1}/{len(todo)} {p['name']}", flush=True)


def run_java(source):
    d = tempfile.mkdtemp()
    try:
        open(os.path.join(d, "Problem.java"), "w").write(source)
        cp = f".{os.pathsep}{JAVATUPLES}"
        try:
            c = subprocess.run(
                ["javac", "-cp", cp, "Problem.java"],
                cwd = d,
                capture_output = True,
                text = True,
                timeout = COMPILE_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return "compile_timeout", ""
        if c.returncode != 0:
            return "compile_error", c.stderr[:500]
        try:
            r = subprocess.run(
                ["java", "-ea", "-cp", cp, "Problem"],
                cwd = d,
                capture_output = True,
                text = True,
                timeout = RUN_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return "run_timeout", ""
        return ("pass", "") if r.returncode == 0 else ("runtime_fail", r.stderr[:500])
    finally:
        shutil.rmtree(d, ignore_errors=True)

problems =list(ds)
sources = [p["prompt"] + cache[p["name"]] + p["tests"] for p in problems]
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    outcomes = list(ex.map(run_java, sources))
results = [
    {
        "name": p["name"], 
        "status": s, 
        "detail": d, 
        "completion": cache[p["name"]]
    }
    for p, (s, d) in zip(problems, outcomes)
    ]

print(Counter(r["status"] for r in results))


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def description(prompt):
    out  = []
    for l in prompt.splitlines():
        l = l.strip()
        if not l.startswith("//"): continue
        if ">>>" in l: break                                
        out.append(l)
    return norm(" ".join(out))

def docs(path):
    with open(path, errors="ignore") as f:
        if path.endswith(".jsonl"):
            for line in f: yield json.loads(line).get("content", "")
        else:
            yield f.read()


def find_contamination(problems):
    needles = {}
    for p in problems:
        d = description(p["prompt"])
        if len(d) >= 30:
            needles[d[:120]] = p["name"]
    hits = set()
    try:
        import ahocorasick
        A = ahocorasick.Automaton()
        for n, name in needles.items():
            A.add_word(n, name)
        A.make_automaton()
        for path in glob.glob(os.path.join(SHARDS_DIR, "*")):
            for doc in docs(path):
                for _, name in A.iter(norm(doc)): hits.add(name)
    except ImportError:
        for path in glob.glob(os.path.join(SHARDS_DIR, "*")):
            for doc in docs(path):
                nd = norm(doc)
                hits |= {name for n, name in needles.items() if n in nd}
    return hits


shards_ready = os.path.isdir(SHARDS_DIR) and glob.glob(os.path.join(SHARDS_DIR, "*"))
contaminated = find_contamination(problems) if shards_ready else set()


kept = [r for r in results if r["name"] not in contaminated]
passed = sum(r["status"] == "pass" for r in kept)
summary = {
    "model": MODEL_ID,
    "total_problems": len(results),
    "contamination_checked": bool(shards_ready),
    "n_contaminated": len(contaminated),
    "contaminated": sorted(contaminated),
    "scored": len(kept),
    "passed": passed,
    "pass@1_pct": round(100 * passed / len(kept), 2) if kept else None,
    "status_counts": dict(Counter(r["status"] for r in results)),
}
print(json.dumps(summary, indent=2))
json.dump({"summary": summary, "results": results}, open(OUT_FILE, "w"), indent=2)