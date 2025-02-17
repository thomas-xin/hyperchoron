import os
import subprocess
import sys

procs = []
def run_conversion(fi, *fo):
	return subprocess.Popen([sys.executable, "hyperchoron.py", "-i", fi, "-o", *fo])
def wait_procs():
	while len(procs) >= 8:
		try:
			procs[0].wait(timeout=2)
		except subprocess.TimeoutExpired:
			pass
		for i in range(len(procs) - 1, -1, -1):
			if procs[i].poll() is not None:
				procs.pop(i)

OUTPUT_FORMATS = ("litematic", "mcfunction", "nbs")
for fmt in OUTPUT_FORMATS:
	if not os.path.exists(f"examples/{fmt}"):
		os.mkdir(f"examples/{fmt}")
for fn in sorted(os.listdir("examples/midi"), key=lambda fn: (fn.endswith(".zip"), os.path.getsize(f"examples/midi/{fn}")), reverse=True):
	names = [fn.rsplit(".", 1)[0] + "." + fmt for fmt in OUTPUT_FORMATS]
	fi = f"examples/midi/{fn}"
	fo = [f"examples/{fmt}/{n}" for fmt, n in zip(OUTPUT_FORMATS, names)]
	if any(not os.path.exists(f) or not os.path.getsize(f) or os.path.getmtime(f) < os.path.getmtime(fi) for f in fo):
		wait_procs()
		procs.append(run_conversion(fi, *fo))

for proc in procs:
	proc.wait()