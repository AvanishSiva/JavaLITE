from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re, subprocess, tempfile, pathlib, json, sys, csv, datetime

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()

SYSTEM = """You edit one Java file.
SEARCH = a few lines copied unchanged from the ORIGINAL file (before your change).
REPLACE = those same lines after your change.
Change the lines in place. Do not add new methods unless the task asks for one.
Reply with ONLY the blocks, no explanations."""

EXAMPLE_FILE = """public class Price {
    public static double withTax(double amount) {
        return amount * 1.2;
    }

    public static String label(double amount) {
        return "EUR " + amount;
    }
}"""

EXAMPLE_TASK = "Make withTax take the tax rate as a second parameter rate instead of using 1.2."

EXAMPLE_REPLY = """<<<<<<< SEARCH
    public static double withTax(double amount) {
        return amount * 1.2;
    }
=======
    public static double withTax(double amount, double rate) {
        return amount * (1 + rate);
    }
>>>>>>> REPLACE"""


PREFILL = "<<<<<<< SEARCH\n"

BLOCK = re.compile(r"<<<<<<< SEARCH[ \t]*\n(.*?)\n=======[ \t]*\n(.*?)\n?>>>>>>> REPLACE", re.S)
FENCE = re.compile(r"```(?:java)?[ \t]*\n(.*?)```", re.S)
LOOSE_SPAN = re.compile(r"<<<<<<< SEARCH[ \t]*\n(.*?)(?:\n>>>>>>> REPLACE|$)", re.S)
SEPARATOR = re.compile(r"(?m)^=======[ \t]*$")
PLACEHOLDERS = {"exact lines from the file", "replacement lines"}

def user_msg(source, instruction):
    return f"File:\n{source}\n\nTask:\n{instruction}"

def generate(source, instruction):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg(EXAMPLE_FILE, EXAMPLE_TASK)},
            {"role": "assistant", "content": EXAMPLE_REPLY},
            {"role": "user", "content": user_msg(source, instruction)}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + PREFILL
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    out = model.generate(**enc, max_new_tokens=768, do_sample=False)
    return PREFILL + tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

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
        if not search.strip():
            return None, "search_empty"
        hits = source.count(search)
        if hits == 0:
            return None, "search_not_found"
        if hits > 1:
            return None, "search_ambiguous"
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

def err_text(r):
    """javac/java error output for diagnosis (or 'timeout')."""
    if r is None:
        return "timeout"
    return (r.stderr or r.stdout).strip()[:2000]

def score_task(t):
    reply = generate(t["before"], t["instruction"]).replace("\r\n", "\n")
    cls = re.search(r"public\s+(?:final\s+)?class\s+(\w+)", t["before"]).group(1)
    new_src, status = apply_blocks(t["before"], reply)
    mode = "block"
    if new_src is None:
        new_src, mode = whole_file_fallback(reply, cls), "wholefile"
    if new_src is None:
        return status, status, reply, ""
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, f"{cls}.java").write_text(new_src)
        pathlib.Path(d, "Test.java").write_text(t["test"])
        r = run(["javac", f"{cls}.java", "Test.java"], d)
        if r is None or r.returncode != 0:
            return "compile_fail", mode, reply, err_text(r)
        r = run(["java", "-cp", ".", "Test"], d)
        if r is None or r.returncode != 0:
            return "test_fail", mode, reply, err_text(r)
    return "pass", mode, reply, ""

paths = sorted(pathlib.Path("eval/edit_tasks").glob("*.json"))
tasks = [json.loads(p.read_text()) for p in paths]
results = []
with open("edit_outputs.jsonl", "w") as f:
    for t in tasks:
        status, mode, reply, error = score_task(t)
        results.append((status, mode))
        print(t["id"], status, mode)
        if error:  
            print("    ", error.splitlines()[0])
        f.write(json.dumps({"id": t["id"], "type": t["type"], "status": status,
                            "mode": mode, "error": error, "reply": reply}) + "\n")

n = len(tasks)
strict = sum(s == "pass" and m == "block" for s, m in results)
lenient = sum(s == "pass" for s, _ in results)
fmt_ok = sum(m == "block" for _, m in results)
print(f"strict pass (valid blocks):  {strict}/{n} = {strict/n:.3f}")
print(f"lenient pass (+ whole file): {lenient}/{n} = {lenient/n:.3f}")
print(f"format ok:                   {fmt_ok}/{n} = {fmt_ok/n:.3f}")
for s in sorted({s for s, _ in results}):
    print(s, sum(x == s for x, _ in results))

if "--record" in sys.argv:
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    if git("status", "--porcelain", "--untracked-files=no"):
        commit += "-dirty"  # uncommitted changes: the hash doesn't fully describe this run
    out = pathlib.Path("results.csv")
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "commit", "model", "edit_pass_strict", "edit_pass_lenient", "format_ok", "n"])
        w.writerow([datetime.date.today().isoformat(), commit, MODEL,
                    f"{strict/n:.3f}", f"{lenient/n:.3f}", f"{fmt_ok/n:.3f}", n])
    print(f"recorded to results.csv (commit {commit})")