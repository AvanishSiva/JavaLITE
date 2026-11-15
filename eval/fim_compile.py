import argparse, csv, json, random, subprocess, tempfile, datetime

from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava


PARSER = Parser(Language(tsjava.language()))
TYPE_NODES = {
    "class_declaration", 
    "interface_declaration",
    "enum_declaration",
    "record_declaration"
}

MAX_CHARACTER = 3000
FIM = "<|fim_prefix|>{p}<|fim_suffix|>{s}<|fim_middle|>"

def compiles(src: str, name : str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / f"{name}.java"
        f.write_text(src)
        try:
            r = subprocess.run(["javac", "-proc:none", "-d", d, str(f)],capture_output=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0

def walk(node):
    yield node
    for c in node.children:
        yield from walk(c)

def main_type_name(root, src: bytes):
    names = []
    for c in root.children:
        if c.type in TYPE_NODES:
            name = src[c.child_by_field_name("name").start_byte:
                       c.child_by_field_name("name").end_byte].decode()
            if b"public" in src[c.start_byte:c.start_byte + 20]:
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
            if b is not None and b.end_byte - b.start_byte > 40: 
                bodies.append(b)
    return rng.choice(bodies) if bodies else None

def build(args):
    rng = random.Random(args.seed)
    files = sorted(Path(args.src_dir).rglob("*.java"))
    rng.shuffle(files)
    cases = []
    for path in files:
        if len(cases) >= args.n:
            break
        try:
            src = path.read_bytes()
            if len(src) > MAX_CHARACTER:
                continue
            root = PARSER.parse(src).root_node
            if root.has_error or not only_java_imports(root, src):
                continue
            name = main_type_name(root, src)
            body = pick_method_body(root, rng)
            if not name or body is None:
                continue
            if not compiles(src.decode(), name):     
                continue
            cases.append({
                "file": str(path), "name": name,
                "prefix": src[:body.start_byte + 1].decode(),   
                "suffix": src[body.end_byte - 1:].decode(),    
                "original": src[body.start_byte + 1:body.end_byte - 1].decode(),
            })
        except Exception:
            continue
    Path(args.cases).write_text("\n".join(json.dumps(c) for c in cases))
    print(f"kept {len(cases)} cases -> {args.cases}")

def score(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16).cuda().eval()

    cases = [json.loads(l) for l in Path(args.cases).read_text().splitlines()]
    ok = 0
    for i, c in enumerate(cases):
        prompt = FIM.format(p=c["prefix"], s=c["suffix"])
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        middle = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        passed = compiles(c["prefix"] + middle + c["suffix"], c["name"])
        ok += passed
        print(f"{i+1}/{len(cases)} compiled={passed}")

    n = len(cases)
    row = [args.label, "fim_compile", n, ok, round(100 * ok / n, 1),
           datetime.date.today().isoformat()]
    with open(args.results, "a", newline="") as f:
        csv.writer(f).writerow(row)
    print("row:", row)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--src_dir", required=True)     
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