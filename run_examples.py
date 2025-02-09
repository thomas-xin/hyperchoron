import os
import subprocess
import sys

procs = []
def run_conversion(fi, fo, fo2):
	return subprocess.Popen([sys.executable, "hyperchoron.py", "-i", fi, "-o", fo, fo2])

if not os.path.exists("examples/litematic"):
	os.mkdir("examples/litematic")
if not os.path.exists("examples/mcfunction"):
	os.mkdir("examples/mcfunction")
for fn in os.listdir("examples/midi"):
	name = fn.rsplit(".", 1)[0] + ".litematic"
	name2 = fn.rsplit(".", 1)[0] + ".mcfunction"
	fi = f"examples/midi/{fn}"
	fo = f"examples/litematic/{name}"
	fo2 = f"examples/mcfunction/{name2}"
	if not os.path.exists(fo) or not os.path.getsize(fo) or os.path.getmtime(fo) < os.path.getmtime(fi):
		while len(procs) >= 8:
			procs.pop(0).wait()
		procs.append(run_conversion(fi, fo, fo2))

for proc in procs:
	proc.wait()