import os
import subprocess
import sys

procs = []
def run_conversion(fi, *fo):
	return subprocess.Popen([sys.executable, "hyperchoron.py", "-i", fi, "-o", *fo])

OUTPUT_FORMATS = ("litematic",)
for fmt in OUTPUT_FORMATS:
	if not os.path.exists(f"examples/{fmt}"):
		os.mkdir(f"examples/{fmt}")
for fn in os.listdir("examples/midi"):
	names = [fn.rsplit(".", 1)[0] + "." + fmt for fmt in OUTPUT_FORMATS]
	fi = f"examples/midi/{fn}"
	fo = [f"examples/{fmt}/{n}" for fmt, n in zip(OUTPUT_FORMATS, names)]
	if any(not os.path.exists(f) or not os.path.getsize(f) or os.path.getmtime(f) < os.path.getmtime(fi) for f in fo):
		while len(procs) >= 8:
			procs.pop(0).wait()
		procs.append(run_conversion(fi, *fo))

for proc in procs:
	proc.wait()