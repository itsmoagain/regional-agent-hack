# Regional Pipeline Demo

Use this notebook to reproduce the CPU-only pipeline for either flagship region:

- `hungary_transdanubia` (winter wheat rotation)
- `jamaica_coffee` (Blue Mountain coffee belt)

Each block mirrors the CLI steps described in the README. Run the notebook twice—once for a baseline tier, once for an optimised tier—and record the CodeCarbon outputs for the hackathon submission.


```python
import subprocess
from pathlib import Path

REGION = 'hungary_transdanubia'  # or 'jamaica_coffee'
DATA_DIR = Path('data') / REGION
CONFIG_PATH = Path('regions/profiles') / f'insight.{REGION}.yml'
assert CONFIG_PATH.exists(), f'Missing config for {REGION}'
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH
```


```python
def run(cmd):
    if isinstance(cmd, (list, tuple)):
        display_cmd = ' '.join(cmd)
    else:
        display_cmd = cmd
    print('↪', display_cmd)
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

run(['python', 'scripts/build_region_cache.py', '--region', REGION])
run(['python', 'scripts/build_region_insights.py', '--region', REGION])
run(['python', 'scripts/train_region_model.py', '--region', REGION, '--tier', '2', '--freq', 'monthly'])
run(['python', 'engine/evaluate_greenai.py', '--region', REGION, '--label', 'baseline', '--command', f'python scripts/train_region_model.py --region {REGION} --tier 1 --freq monthly'])
run(['python', 'engine/evaluate_greenai.py', '--region', REGION, '--label', 'optimised', '--command', f'python scripts/train_region_model.py --region {REGION} --tier 3 --freq monthly'])
run(['python', 'scripts/flag_anomalies.py', '--region', REGION, '--config', str(CONFIG_PATH)])
```
