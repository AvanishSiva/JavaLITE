from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re, subprocess, tempfile, pathlib, json

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()

SYSTEM = """You edit one Java file.
SEARCH = a few lines copied unchanged from the ORIGINAL file (before your change).
REPLACE = those same lines after your change.
Only include the lines that change. Reply with ONLY the blocks, no explanations."""

EXAMPLE_FILE = """public class Counter {
    private int n;

    public void inc() {
        n++;
    }
}"""

EXAMPLE_REPLY = """<<<<<<< SEARCH
    public void inc() {
        n++;
    }
=======
    public void inc() {
        n++;
    }

    public void reset() {
        n = 0;
    }
>>>>>>> REPLACE"""

BLOCK = re.compile(r"<<<<<<< SEARCH[ \t]*\n(.*?)\n=======[ \t]*\n(.*?)\n?>>>>>>> REPLACE", re.S)
FENCE = re.compile(r"```(?:java)?[ \t]*\n(.*?)```", re.S)
LOOSE_SPAN = re.compile(r"<<<<<<< SEARCH[ \t]*\n(.*?)(?:\n>>>>>>> REPLACE|$)", re.S)
SEPARATOR = re.compile(r"(?m)^=======[ \t]*$")
PLACEHOLDERS = {"exact lines from the file", "replacement lines"}

def user_msg(source, instruction):
    return f"File:\n{source}\n\nTask:\n{instruction}"

def generate(source, instruction):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg(EXAMPLE_FILE, "Add a method reset() that sets n back to 0.")},
            {"role": "assistant", "content": EXAMPLE_REPLY},
            {"role": "user", "content": user_msg(source, instruction)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    out = model.generate(**enc, max_new_tokens=768, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

def norm(text):
    """Normalise line endings and strip trailing whitespace on each line."""
    return "\n".join(l.rstrip() for l in text.replace("\r\n", "\n").split("\n"))

def clean(text):
    """Keep the last chunk after any stray '=======' and drop copied placeholder lines."""
    text = SEPARATOR.split(norm(text))[-1]
    return "\n".join(l for l in text.split("\n") if l.strip() not in PLACEHOLDERS).strip("\n")

def apply_blocks(source, reply):
    blocks = BLOCK.findall(reply)
    if not blocks:
        return None, "no_block"
    source = norm(source)
    for search, rpl in blocks:
        search = norm(search)
        if not search.strip() or source.count(search) != 1:
            return None, "search_not_found_or_ambiguous"
        source = source.replace(search, clean(rpl))
    return source, "ok"

def whole_file_fallback(reply, cls):
    """Use the full edited file if the model wrote one (in a ``` fence or a broken block)."""
    for c in reversed(FENCE.findall(reply) + LOOSE_SPAN.findall(reply)):
        chunk = clean(c)
        if re.search(rf"\bclass\s+{re.escape(cls)}\b", chunk):
            return chunk + "\n"
    return None

def run(cmd, cwd):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None

def score_task(t):
    reply = generate(t["before"], t["instruction"]).replace("\r\n", "\n")
    cls = re.search(r"public\s+(?:final\s+)?class\s+(\w+)", t["before"]).group(1)
    new_src, status = apply_blocks(t["before"], reply)
    mode = "block"
    if new_src is None:
        new_src, mode = whole_file_fallback(reply, cls), "wholefile"
    if new_src is None:
        return status, status, reply
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, f"{cls}.java").write_text(new_src)
        pathlib.Path(d, "Test.java").write_text(t["test"])
        r = run(["javac", f"{cls}.java", "Test.java"], d)
        if r is None or r.returncode != 0:
            return "compile_fail", mode, reply
        r = run(["java", "-cp", ".", "Test"], d)
        if r is None or r.returncode != 0:
            return "test_fail", mode, reply
    return "pass", mode, reply

paths = sorted(pathlib.Path("eval/edit_tasks").glob("*.json"))
tasks = [json.loads(p.read_text()) for p in paths]
results = []
with open("edit_outputs.jsonl", "w") as f:
    for t in tasks:
        status, mode, reply = score_task(t)
        results.append((status, mode))
        print(t["id"], status, mode)
        f.write(json.dumps({"id": t["id"], "type": t["type"], "status": status,
                            "mode": mode, "reply": reply}) + "\n")

n = len(tasks)
strict = sum(s == "pass" and m == "block" for s, m in results)
lenient = sum(s == "pass" for s, _ in results)
fmt_ok = sum(m == "block" for _, m in results)
print(f"strict pass (valid blocks):  {strict}/{n} = {strict/n:.3f}")
print(f"lenient pass (+ whole file): {lenient}/{n} = {lenient/n:.3f}")
print(f"format ok:                   {fmt_ok}/{n} = {fmt_ok/n:.3f}")
for s in sorted({s for s, _ in results}):
    print(s, sum(x == s for x, _ in results))