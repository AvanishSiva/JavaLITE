from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re, subprocess, tempfile, pathlib, json

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()

SYSTEM = """You edit one Java file. Reply with ONLY one or more blocks in this format: <<<<<<< SEARCH exact lines from the file ======= replacement lines >>>>>>> REPLACE The SEARCH text must match the file exactly."""

BLOCK = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n?>>>>>>> REPLACE", re.S)

def generate(source, instruction):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"File:\n{source}\n\nTask:\n{instruction}"}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    out = model.generate(**enc, max_new_tokens=512, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

def apply_blocks(source, reply):
    blocks = BLOCK.findall(reply)
    if not blocks:
        return None, "no_block"
    for search, rpl in blocks:
        if source.count(search) != 1:
            return None, "search_not_found_or_ambiguous"
        source = source.replace(search, rpl)
    return source, "ok"

def run(cmd, cwd):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None

def score_task(t):
    reply = generate(t["before"], t["instruction"])
    new_src, status = apply_blocks(t["before"], reply)
    if new_src is None:
        return status, reply
    cls = re.search(r"public\s+(?:final\s+)?class\s+(\w+)", t["before"]).group(1)
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, f"{cls}.java").write_text(new_src)
        pathlib.Path(d, "Test.java").write_text(t["test"])
        r = run(["javac", f"{cls}.java", "Test.java"], d)
        if r is None or r.returncode != 0:
            return "compile_fail", reply
        r = run(["java", "-cp", ".", "Test"], d)
        if r is None or r.returncode != 0:
            return "test_fail", reply
    return "pass", reply

paths = sorted(pathlib.Path("eval/edit_tasks").glob("*.json"))
tasks = [json.loads(p.read_text()) for p in paths]
results = []
with open("edit_outputs.jsonl", "w") as f:
    for t in tasks:
        status, reply = score_task(t)
        results.append(status)
        print(t["id"], status)
        f.write(json.dumps({"id": t["id"], "type": t["type"],
                            "status": status, "reply": reply}) + "\n")

passed = results.count("pass")
print(f"{passed}/{len(tasks)} = {passed/len(tasks):.3f}")
for s in sorted(set(results)):
    print(s, results.count(s))