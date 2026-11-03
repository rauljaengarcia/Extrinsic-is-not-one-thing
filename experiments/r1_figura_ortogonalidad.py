#!/usr/bin/env python3
r"""
R1.Q3 (Revisor 1): la Figura 2 del manuscrito (fig1_ortogonalidad) con intervalos
EMPAREJADOS.

Regenera la figura con el mismo codigo que la publico, `fig_ortogonalidad()` de
experiments/figuras.py, sin tocarlo: solo le da otros datos.  Cambian los intervalos de
los paneles (b) y (c), que pasan a ser los emparejados de results/r1_bootstrap.json
(Tabla 4: retirada Phi_R/Phi_B y reetiquetado L1/L0).  Los puntos son los mismos con y
sin emparejar, y se comprueba antes de dibujar.

    python experiments/r1_figura_ortogonalidad.py
        -> figures/r1_fig1_ortogonalidad_emparejada.png y .pdf  (no pisa la publicada,
           figures/fig1_ortogonalidad.png; el manuscrito revisado la incluye con ese
           nombre: README_R1.md, nota 7)
    python experiments/r1_figura_ortogonalidad.py --publicada DIR
        regenera ademas, en DIR, la version publicada (sin emparejar), para comparar
"""
import argparse
import copy
import shutil
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "experiments"))

import runner                                          # noqa: E402
import r1_comun as rc                                  # noqa: E402

CONDS = ("Ix", "Ip", "Ex", "Ep")
#: nombre de la salida en figures/: no el de la publicada, fig1_ortogonalidad, que se
#: pisaria al copiar codigo_R1/ en el repositorio (README_R1.md, nota 7)
SALIDA = "r1_fig1_ortogonalidad_emparejada"


def emparejados(orto, boot):
    """ortogonalidad.json con los intervalos de la Tabla 4 cambiados por los emparejados."""
    d = copy.deepcopy(orto)
    res = d["manifiesto"]["resumen"]
    t4 = boot["manifiesto"]["tabla4"]
    for c in CONDS:
        for eje, clave in (("eje_C", "retirada"), ("eje_A", "reetiquetado")):
            par = t4[c][clave]["emparejado"]
            if abs(par[0] - res[eje][c]["ratio"][0]) > 1e-12:
                sys.exit(f"{c} {clave}: el punto emparejado no es el publicado")
            res[eje][c]["ratio"] = list(par)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publicada", default=None, metavar="DIR")
    args = ap.parse_args()
    rc.preparar()          # runner.FIGURAS -> figures/ antes de importar figuras.py
    import figuras         # noqa: E402  (aplica el estilo de src/estilo.py)
    orto = runner.load("ortogonalidad")
    if args.publicada:
        figuras.FIGURAS = Path(args.publicada)
        figuras.FIGURAS.mkdir(parents=True, exist_ok=True)
        figuras.load = lambda nombre: orto
        figuras.fig_ortogonalidad()
    datos = emparejados(orto, runner.load("r1_bootstrap"))
    figuras.load = lambda nombre: datos
    # fig_ortogonalidad() escribe fig1_ortogonalidad.{png,pdf}, el nombre de la publicada:
    # se escribe en una carpeta temporal y se copia a figures/ con el de la R1
    with tempfile.TemporaryDirectory() as tmp:
        figuras.FIGURAS = Path(tmp)
        figuras.fig_ortogonalidad()
        for ext in (".png", ".pdf"):
            shutil.copyfile(Path(tmp) / f"fig1_ortogonalidad{ext}", rc.FIGURES / f"{SALIDA}{ext}")
    print(f"  copiada a figures/ como {SALIDA}.png + {SALIDA}.pdf")
    for c in CONDS:
        r = datos["manifiesto"]["resumen"]
        print(f"  {c}: retirada {rc.fmt(r['eje_C'][c]['ratio'])}   "
              f"reetiquetado {rc.fmt(r['eje_A'][c]['ratio'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
