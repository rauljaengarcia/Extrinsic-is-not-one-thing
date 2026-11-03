#!/usr/bin/env python3
r"""
R2.1 (Revisor 2): la topologia de la asignacion, con los datos que ya existen.

Lee results/ortogonalidad.json, las filas de reetiquetado (`{cond}_iny`: 100
semillas, 150 clones inyectados en t = 10 000, ventana de acceso
[10 000, 10 600)), y mide por condicion, con intervalo percentil emparejado por
semilla (10 000 replicas, BOOT_SEED; todas las medidas de una condicion usan
las mismas replicas):

  kappa        = from_L0_pc_L1 / from_L0_pc_L0
                 asignacion per capita de los donantes residentes (L0) a los
                 reetiquetados, sobre la que dan a los residentes.  Es el
                 cociente L1/L0 de la Tabla 4.
  s            = recv_pc_L1 / recv_pc_L0
                 suministro total per capita a los reetiquetados, de cualquier
                 origen, sobre el de los residentes.
  flow_ij      energia de donantes con etiqueta i a receptores con etiqueta j,
               en la ventana (la matriz 2x2 de flujos).
  nacimientos  = born_pc_L1 / born_pc_L0, por organismo-paso.

Cocientes de medias, como la Tabla 4, no medias de cocientes.  Lo esperado,
que aqui se confirma o no: x -> kappa ~ 0.99 y s ~ 1.00; I/p (particion) ->
kappa = 0 y s ~ 1.00; E/p (exclusion) -> kappa = 0 y s = 0.  Los dos ceros de
kappa son exactos y la s los separa: es la diferencia entre excluir y partir
que la §6.4 senala sin medirla.

    python experiments/r1_topologia.py       # no simula; escribe results/r1_topologia.json
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "experiments"))

import runner                                          # noqa: E402
import r1_comun as rc                                  # noqa: E402
import estadistica as st                               # noqa: E402
import estadistica_r1 as er                            # noqa: E402
from runner import col, banner                         # noqa: E402

CONDS = ("Ix", "Ip", "Ex", "Ep")
ETIQ = {"Ix": "I/x", "Ip": "I/p", "Ex": "E/x", "Ep": "E/p"}
#: (denominador, numerador) de cada cociente
COCIENTES = {"kappa": ("from_L0_pc_L0", "from_L0_pc_L1"),
             "s": ("recv_pc_L0", "recv_pc_L1"),
             "nacimientos": ("born_pc_L0", "born_pc_L1")}
FLUJOS = ("flow_00", "flow_01", "flow_10", "flow_11")
#: lo esperado segun el encargo, con la tolerancia de "~"
ESPERADO = {"x": {"kappa": 0.99, "s": 1.00}, "Ip": {"kappa": 0.0, "s": 1.00},
            "Ep": {"kappa": 0.0, "s": 0.0}}
TOL = 0.01


def medir(filas):
    """kappa, s, flujos y nacimientos de un conjunto de filas de reetiquetado."""
    filas = sorted(filas, key=lambda r: r["seed"])
    out = {"n_semillas": len(filas),
           "semillas": [filas[0]["seed"], filas[-1]["seed"]] if filas else None}
    for nombre, (den, num) in COCIENTES.items():
        out[nombre] = rc.iv(er.percentil_pareado(er.cociente, col(filas, den),
                                                 col(filas, num)))
    for f in FLUJOS:
        out[f] = rc.iv(er.percentil_pareado(lambda m: m, col(filas, f)))
    for k in ("recv_pc_L0", "recv_pc_L1", "from_L0_pc_L0", "from_L0_pc_L1",
              "born_pc_L0", "born_pc_L1"):
        out[f"media_{k}"] = float(col(filas, k).mean())
    return out


def esperado(cond, m):
    clave = "x" if cond[1] == "x" else cond
    exp = ESPERADO[clave]
    res = {}
    for k, v in exp.items():
        obs = m[k][0]
        res[k] = {"esperado": v, "obtenido": obs,
                  "confirma": bool(obs == v) if v == 0.0 else bool(abs(obs - v) <= TOL)}
    return res


def main():
    rc.preparar()
    orto = rc.cargar("ortogonalidad")
    banner("R2.1 -- topologia de la asignacion (ortogonalidad.json, reetiquetado)")
    filas_out = []
    print(f"  {'cond':<5} {'kappa':>24} {'s':>24} {'nacimientos L1/L0':>24}")
    for c in CONDS:
        sel = [r for r in orto["filas"] if r["label"] == f"{c}_iny"]
        m = medir(sel)
        m.update(cond=c, etiqueta=ETIQ[c], esperado=esperado(c, m))
        filas_out.append(m)
        print(f"  {ETIQ[c]:<5} {rc.fmt(m['kappa']):>24} {rc.fmt(m['s']):>24} "
              f"{rc.fmt(m['nacimientos']):>24}")
    print(f"\n  {'cond':<5} " + " ".join(f"{f:>26}" for f in FLUJOS))
    for m in filas_out:
        print(f"  {m['etiqueta']:<5} " + " ".join(
            f"{rc.fmt(m[f], 0):>26}" for f in FLUJOS))
    print("\n  esperado (tolerancia +-0.01 donde dice ~; exacto donde dice =):")
    for m in filas_out:
        for k, e in m["esperado"].items():
            print(f"    {m['etiqueta']:<5} {k:<6} esperado {e['esperado']:.2f}  "
                  f"obtenido {e['obtenido']:.4f}  -> {'SI' if e['confirma'] else 'NO'}")
    # kappa es el cociente L1/L0 de la Tabla 4: mismo punto que el publicado
    pub = orto["manifiesto"]["resumen"]["eje_A"]
    iguales = all(abs(m["kappa"][0] - pub[m["cond"]]["ratio"][0]) < 1e-12 for m in filas_out)
    print(f"\n  kappa == punto L1/L0 publicado de la Tabla 4 en las cuatro: {iguales}")

    runner.save("r1_topologia", filas_out, [], extra={
        "n_simulaciones": 0, "fuente": "ortogonalidad.json (filas {cond}_iny)",
        "bootstrap": {"tipo": "percentil emparejado por semilla", "n_boot": st.N_BOOT,
                      "semilla": st.BOOT_SEED},
        "kappa_igual_a_tabla4": iguales,
        "procedencia": rc.procedencia(__file__, ["ortogonalidad.json"]),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
