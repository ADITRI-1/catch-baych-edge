"""Package only evidence from a successful GPU run; standard library only."""
from pathlib import Path
import csv
import zipfile

root = Path(__file__).resolve().parents[1]
results = root / 'results'
log = (results / 'run.log').read_text()
if 'Validation: PASS' not in log:
    raise SystemExit('A successful actual GPU run is required.')
with (results / 'metrics.csv').open() as f:
    rows = list(csv.DictReader(f))
if sum(int(row['count']) for row in rows) < 100:
    raise SystemExit('Run at least 100 images for this evidence package.')
if any(int(row['mismatches']) for row in rows):
    raise SystemExit('GPU validation failed.')
with zipfile.ZipFile(root / 'execution_evidence.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for file in sorted(results.iterdir()):
        if file.is_file():
            z.write(file, 'results/' + file.name)
print('Created execution_evidence.zip from actual run artifacts.')
