import csv
import glob
import sys

pattern = sys.argv[1] if len(sys.argv) > 1 else "runs/*_scfd_default/energy.csv"
paths = sorted(glob.glob(pattern))
if not paths:
    print(f"No files matched pattern: {pattern}")
    sys.exit(1)
path = paths[-1]
with open(path, newline="") as fh:
    rows = list(csv.DictReader(fh))
start = float(rows[0]["total"])
end = float(rows[-1]["total"])
steps = len(rows)
absolute = end - start
relative = absolute / start if start else float("inf")
print(f"Energy start {start:.6e}, end {end:.6e}, absolute change {absolute:.6e}, relative change {relative:.4%}, steps {steps}, file {path}")
