#!/usr/bin/env bash
set -euo pipefail
unset PYTHONOPTIMIZE
export PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
# Bootstrap only stdlib after authenticating the entire frozen launch plan.
exec python3 - "$@" <<'PY'
import hashlib, json, pathlib, subprocess, sys
plan, expected, step, target = sys.argv[1:]
path = pathlib.Path(plan)
if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
    raise SystemExit('Launch plan hash changed')
data = json.loads(path.read_text())
for name in ('run_script', 'wrapper'):
    script = pathlib.Path(data[name])
    if hashlib.sha256(script.read_bytes()).hexdigest() != data['pins'][str(script)]:
        raise SystemExit('Bootstrap script hash changed')
subprocess.run([sys.executable, data['run_script'], plan, expected, step, target], check=True)
PY
