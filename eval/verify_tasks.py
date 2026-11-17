
import json, glob, os, re, subprocess, sys, tempfile

def run(src, test):
    name = re.search(r"public\s+(?:final\s+)?(?:class|record|enum|interface)\s+(\w+)", src).group(1)
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/{name}.java", "w").write(src)
        open(f"{d}/Test.java", "w").write(test)
        if subprocess.run(["javac", "--release", "17", f"{name}.java", "Test.java"], cwd=d, capture_output=True).returncode:
            return False
        r = subprocess.run(["java", "-cp", ".", "Test"], cwd=d, capture_output=True, text=True, timeout=30)
        lines = r.stdout.strip().splitlines()
        return r.returncode == 0 and bool(lines) and lines[-1] == "PASS"

bad = 0
for p in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else "eval/edit_tasks", "*.json"))):
    t = json.load(open(p))
    ok = not run(t["before"], t["test"]) and run(t["after"], t["test"])
    bad += not ok
    print(("ok   " if ok else "BAD  ") + t["id"])
print(f"{bad} bad task(s)")
sys.exit(1 if bad else 0)
