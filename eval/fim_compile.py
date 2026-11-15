import argparse, csv, json, random, subprocess, tempfile, datetime, difflib
from collections import Counter
from pathlib import Path

import pandas as pd
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava


PARSER = Parser(Language(tsjava.language()))
TYPE_NODES = {
    "class_declaration", 
    "interface_declaration",
    "enum_declaration",
    "record_declaration"
}

MAX_BYTES = 10000        # max file size (bytes)
MAX_BODY_CHARS = 700     # ~200 tokens, so the answer fits in max_new_tokens
MIN_BODY_BYTES = 40
MAX_NEW_TOKENS = 256
FIM = "<|fim_prefix|>{p}<|fim_suffix|>{s}<|fim_middle|>"

def compiles(src: str, name: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / f"{name}.java"
        f.write_text(src, encoding="utf-8")
        try:
            r = subprocess.run(
                ["javac", "-encoding", "UTF-8", "-proc:none", "-d", d, str(f)],
                capture_output=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0


def walk(node):
    yield node
    for c in node.children:
        yield from walk(c)


def is_public(type_node) -> bool:
    for c in type_node.children:
        if c.type == "modifiers":
            return any(m.type == "public" for m in c.children)
    return False


def main_type_name(root, src: bytes):
    names = []
    for c in root.children:
        if c.type in TYPE_NODES:
            n = c.child_by_field_name("name")
            name = src[n.start_byte:n.end_byte].decode()
            if is_public(c):
                return name
            names.append(name)
    return names[0] if names else None


def only_java_imports(root, src: bytes) -> bool:
    for c in root.children:
        if c.type == "import_declaration":
            t = src[c.start_byte:c.end_byte].decode()
            if not (t.startswith("import java.") or t.startswith("import static java.")):
                return False
    return True


def pick_method_body(root, rng):
    bodies = []
    for n in walk(root):
        if n.type == "method_declaration":
            b = n.child_by_field_name("body")
            if b is None:
                continue
            size = b.end_byte - b.start_byte
            if MIN_BODY_BYTES < size <= MAX_BODY_CHARS:
                bodies.append(b)
    return rng.choice(bodies) if bodies else None


# ---------- build ----------

def build(args):
    rng = random.Random(args.seed)
    df = pd.read_parquet(args.src)
    if args.code_col not in df.columns:
        raise SystemExit(f"column '{args.code_col}' not found. columns: {df.columns.tolist()}")

    rows = list(df[args.code_col].items())
    rng.shuffle(rows)

    cases, drops = [], Counter()
    for idx, code in rows:
        if len(cases) >= args.n:
            break
        try:
            if code is None:
                drops["empty"] += 1; continue
            src = code.encode("utf-8") if isinstance(code, str) else bytes(code)
            if len(src) > MAX_BYTES:
                drops["too_big"] += 1; continue
            root = PARSER.parse(src).root_node
            if root.has_error:
                drops["parse_error"] += 1; continue
            if not only_java_imports(root, src):
                drops["non_java_import"] += 1; continue
            name = main_type_name(root, src)
            if not name:
                drops["no_type"] += 1; continue
            body = pick_method_body(root, rng)
            if body is None:
                drops["no_body"] += 1; continue
            if not compiles(src.decode("utf-8"), name):
                drops["no_compile"] += 1; continue
            cases.append({
                "file": f"row{idx}", "name": name,
                "prefix": src[:body.start_byte + 1].decode("utf-8"),
                "suffix": src[body.end_byte - 1:].decode("utf-8"),
                "original": src[body.start_byte + 1:body.end_byte - 1].decode("utf-8"),
            })
        except Exception as e:
            key = f"exc:{type(e).__name__}"
            drops[key] += 1
            if drops[key] == 1:
                print(f"row{idx}: {e!r}")

    out = Path(args.cases)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(c) for c in cases))
    print(f"rows: {len(rows)}  drops: {dict(drops)}")
    print(f"kept {len(cases)} cases -> {out}")


def cut_at_special(gen_ids, special_ids):
    """Keep tokens up to (not including) the first special/EOS token."""
    out = []
    for t in gen_ids:
        if t in special_ids:
            break
        out.append(t)
    return out


def score(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    lines = Path(args.cases).read_text().splitlines()
    cases = [json.loads(l) for l in lines if l.strip()]
    n = len(cases)
    if n == 0:
        raise SystemExit("no cases - run build first")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    special_ids = set(tok.all_special_ids)

    ok = baseline_ok = exact = 0
    sim_total = 0.0
    for i, c in enumerate(cases):
        # empty-middle baseline: how many pass for free?
        baseline_ok += compiles(c["prefix"] + c["suffix"], c["name"])

        prompt = FIM.format(p=c["prefix"], s=c["suffix"])
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = out[0, enc.input_ids.shape[1]:].tolist()
        middle = tok.decode(cut_at_special(gen, special_ids))

        passed = compiles(c["prefix"] + middle + c["suffix"], c["name"])
        ok += passed
        exact += middle.strip() == c["original"].strip()
        sim_total += difflib.SequenceMatcher(None, middle.strip(), c["original"].strip()).ratio()
        print(f"{i+1}/{n} compiled={passed}")

    date = datetime.date.today().isoformat()
    rows = [
        [args.label, "fim_compile", n, ok, round(100 * ok / n, 1), date],
        [args.label, "fim_exact_match", n, exact, round(100 * exact / n, 1), date],
        [args.label, "fim_edit_sim", n, "", round(100 * sim_total / n, 1), date],
        ["empty_baseline", "fim_compile", n, baseline_ok, round(100 * baseline_ok / n, 1), date],
    ]
    res = Path(args.results)
    new_file = not res.exists()
    with open(res, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["label", "metric", "n", "passed", "score", "date"])
        w.writerows(rows)
    for r in rows:
        print("row:", r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--src", required=True, help="parquet file with Java source")
    b.add_argument("--code_col", default="content")
    b.add_argument("--cases", default="eval/fim_cases.jsonl")
    b.add_argument("--n", type=int, default=300)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=build)

    s = sub.add_parser("score")
    s.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B")
    s.add_argument("--label", default="Qwen2.5-Coder-0.5B")
    s.add_argument("--cases", default="eval/fim_cases.jsonl")
    s.add_argument("--results", default="results.csv")
    s.set_defaults(fn=score)

    a = ap.parse_args()
    a.fn(a)