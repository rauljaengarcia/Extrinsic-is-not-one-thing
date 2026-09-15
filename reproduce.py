#!/usr/bin/env python3
"""
Single entry point for reproducing every published result.

    python reproduce.py tests          # regression suite, ~2 min
    python reproduce.py validation     # substrate validation      -> validacion.json
    python reproduce.py orthogonality  # the two interventions     -> ortogonalidad.json
    python reproduce.py arms           # three arms, 100 seeds     -> brazos.json
    python reproduce.py c1             # emergent partiality       -> c1_*.json
    python reproduce.py c2             # V_k and N_e               -> c2_ne.json
    python reproduce.py figures        # figures from existing JSON
    python reproduce.py all            # everything, in order

WHY THIS FILE EXISTS, beyond convenience.

`src/runner.py` derives its output directories from its own location, as
`<repo>/resultados` and `<repo>/figuras`.  Those names cannot be changed in
place: `runner.code_hash()` digests every file in `src/`, and that digest is
recorded as `code_sha256_16` in the manifest of every file in `results/`.
Editing `src/runner.py` to rename a directory would silently invalidate the
provenance record of all six published result files, for no scientific gain.

So the directories keep their published names in `src/`, and this script
redirects them to the repository's English-named `results/` and `figures/`
before anything is imported.  `src/` stays byte-identical to the tree that
produced the published numbers, and its hash still verifies.
"""
import runpy
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tests"))
sys.path.insert(0, str(RAIZ / "experiments"))

RESULTS = RAIZ / "results"
FIGURES = RAIZ / "figures"

PASOS = {
    "validation":    ("experiments/validacion.py",       "~8 min"),
    "orthogonality": ("experiments/exp_ortogonalidad.py", "~4 min"),
    "arms":          ("experiments/exp_brazos.py",        "~12 min"),
    "c1":            ("experiments/analisis_c1.py",       "~3 min"),
    "c2":            ("experiments/analisis_c2.py",       "~6 min"),
    "figures":       ("experiments/figuras.py",           "~20 s"),
    "tests":         ("tests/tests.py",                   "~2 min"),
}

ORDEN = ["tests", "validation", "orthogonality", "arms", "c1", "c2", "figures"]


def preparar():
    """Redirige las salidas de runner ANTES de que nadie importe nada de el."""
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    import runner
    runner.RESULTADOS = RESULTS
    runner.FIGURAS = FIGURES
    return runner


def correr(nombre):
    guion, _ = PASOS[nombre]
    print(f"\n{'=' * 70}\n  {nombre}  ->  {guion}\n{'=' * 70}")
    preparar()
    runpy.run_path(str(RAIZ / guion), run_name="__main__")


def uso():
    print(__doc__)
    print("  paso             fichero                          duracion aprox.")
    print("  " + "-" * 66)
    for k in ORDEN:
        g, d = PASOS[k]
        print(f"  {k:<16} {g:<32} {d}")
    return 2


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(uso())
    arg = sys.argv[1]
    if arg == "all":
        for k in ORDEN:
            correr(k)
    elif arg in PASOS:
        correr(arg)
    else:
        print(f"paso desconocido: {arg}\n")
        sys.exit(uso())
