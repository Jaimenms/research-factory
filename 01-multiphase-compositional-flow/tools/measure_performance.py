"""Tempos de desempenho do volume I (uso: uv run python tools/measure_performance.py): primeira corrida (com compilação JIT) e segunda (compilada) por
modelo × solver no sistema offshore, NumPy de referência, e custo por amostra da marcha vetorizada.
Usa só a API pública do simulador. Grava runs/performance_compiled/timings.json."""
import json, time
from pathlib import Path
import numpy as np
from cmfsim.io.spec import Case, load_json
from cmfsim.simulation import simulate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "performance_compiled" / "timings.json"
batch = json.load(open(ROOT / "batches" / "solver_matrix.json"))
res = json.loads(OUT.read_text()) if OUT.exists() else {}
for run in batch["runs"]:
    name = run["name"]
    if not name.startswith("offshore_") or name in res:
        continue
    backend = run.get("backend", "jax")
    d = load_json(ROOT / run["case"], run.get("set"))
    case = Case.from_dict(d)
    entry = {"backend": backend}
    t0 = time.perf_counter(); r1 = simulate(case, backend=backend, progress=False); entry["first_wall"] = time.perf_counter() - t0
    entry["first_cpu"] = r1.info.get("cpu")
    if backend == "jax":
        t0 = time.perf_counter(); r2 = simulate(case, backend=backend, progress=False); entry["second_wall"] = time.perf_counter() - t0
        entry["second_cpu"] = r2.info.get("cpu")
        entry["max_dP_Pa"] = float(np.max(np.abs(r1.P - r2.P)))
    entry["iterations"] = r1.info.get("iterations"); entry["n_accepted"] = r1.info.get("n_accepted"); entry["n_nodes"] = r1.info.get("n_nodes")
    res[name] = entry
    OUT.write_text(json.dumps(res, indent=1))
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in entry.items()}, flush=True)
# marcha vetorizada: 300 amostras do poço, primeira chamada e segunda (compilada)
if "vmap_well_300" not in res:
    from surrogate.dataset import sample_inputs, simulate_outputs, thesis_ranges
    case = Case.from_dict(load_json(ROOT / "cases" / "surrogate" / "well.json"))
    X = sample_inputs(thesis_ranges("well"), 300, seed=0)
    t0 = time.perf_counter(); simulate_outputs(case, X, chunk=100); t1 = time.perf_counter() - t0
    t0 = time.perf_counter(); simulate_outputs(case, X, chunk=100); t2 = time.perf_counter() - t0
    res["vmap_well_300"] = {"first_wall": t1, "second_wall": t2, "n": 300}
    OUT.write_text(json.dumps(res, indent=1))
    print("vmap_well_300", round(t1, 2), round(t2, 2), flush=True)
