#!/usr/bin/env python3
"""
LOS TRES BRAZOS — estimacion de rho, con su calibrado y su guardian.

    B  linea base    el soporte tal como se suministra
    R  retirada      el soporte retirado por completo
    S  sustitucion   retirado, y un proveedor de la misma clase de recurso
                     accesible bajo una regla exchangeable, IGUALADA a B en
                     toda propiedad causal salvo la identidad

    delta = (Phi_B - Phi_R) / max(Phi_B, Phi_R)
    rho   = (Phi_B - Phi_S) / (Phi_B - Phi_R),   NA donde el guardian muerde

B y R juntos son el diseno de retirada de andamio que ya existe en la
literatura.  **El brazo S es la aportacion**: separa "hacia falta el recurso"
de "hacia falta que alguien decidiera darselo a ESTA poblacion".

Tres manipulaciones, con `rho` esperado en NA, bajo y alto.  Con un solo punto
`rho` seria un recurso retorico; con los tres extremos es una medida.

El calibrado del igualado se hace sobre semillas piloto DISJUNTAS de las
confirmatorias y se descarta.  Ver `PREREGISTRO.md`.

    python exp_brazos.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import runner                                          # noqa: E402
import numpy as np                                     # noqa: E402
from runner import spec, run_grid, save, col, banner   # noqa: E402
import estadistica as st                               # noqa: E402

# Escala del experimento.  Reducible por entorno para una prueba de humo; los
# valores por defecto son los del pre-registro.
T = int(os.environ.get("EXP_T", 20_000))
N_SEED = int(os.environ.get("EXP_SEEDS", 30))      # confirmatorias: 0..N-1
_NP = int(os.environ.get("EXP_PILOTOS", 12))
SEEDS_PILOTO = range(1000, 1000 + _NP)             # disjuntas; se descartan
N0 = 400
DOSIS = (0, 40, 100, 200)       # tamano del linaje competidor L1
TOL = 0.05                      # tolerancia de igualado pre-registrada
EPS_FRAC = 0.05                 # guardian
#: Intensidades del sustituto barridas en la manipulacion de supresion.  No se
#: calibra un punto porque la cantidad sobre la que habria que igualar resulta
#: casi insensible a la manipulacion; se informa rho en todo el rango.
FRACS_S = (0.05, 0.15, 0.30, 0.45, 0.60)

PHI = "phi_focal"               # Phi primaria: ciclos autonomos del linaje focal


# ==========================================================================
# Calibrado del igualado — sobre pilotos que se descartan
# ==========================================================================

def _reproduccion_total(rows):
    """Reproduccion total realizada en el mundo.

    Es la cantidad sobre la que hay que igualar los brazos de la manipulacion
    de supresion, y llegar a ella costo dos intentos fallidos que conviene
    dejar escritos:

    1. Igualar `suppressed / (suppressed + phi_focal)` mezcla un numerador de
       eventos de L1 con un denominador de nacimientos de L0. No es una
       fraccion de nada.
    2. Igualar `suppressed / (suppressed + births)` tampoco vale, porque
       **`suppressed` cuenta organismo-pasos bloqueados, no eventos
       impedidos**: un individuo vetado sigue siendo elegible al paso
       siguiente y se vuelve a contar. En el brazo parcial los miembros de L1
       quedan vetados para siempre y se acumulan sin limite; en el brazo
       abierto el vetado de un paso puede no serlo al siguiente. Los dos
       brazos se inflan, y con factores distintos, asi que el cociente no es
       comparable entre ellos.

    Los nacimientos realizados no tienen ese problema: cuentan eventos, una
    vez cada uno, y son la consecuencia operativa de «se impide la misma
    cantidad de reproduccion».
    """
    return float(col(rows, "births").mean())


def calibrar_supresion(n0_L1, iters=9):
    """Iguala la FRACCION VETADA del brazo S a la del brazo B, por biseccion.

    En B se veta a todo L1; la fraccion vetada entre los elegibles es un
    resultado emergente de la demografia, no un parametro. Hay que medirla en
    pilotos y despues buscar el `suppress_frac` del brazo S que la reproduce.

    Se usa biseccion y no una rejilla: la rejilla gruesa de la primera version
    no bajaba lo suficiente ni afinaba bastante, y dejaba el igualado fuera de
    la tolerancia pre-registrada, que es tanto como no estimar rho.
    """
    base = {"n0_L1": n0_L1}
    rowsB = run_grid([spec(cond="Ex", seed=s, T=T,
                           cfg={**base, "suppress": "partial"})
                      for s in SEEDS_PILOTO], quiet=True)
    rows0 = run_grid([spec(cond="Ex", seed=s, T=T,
                           cfg={**base, "suppress": "none"})
                      for s in SEEDS_PILOTO], quiet=True)
    obj = _reproduccion_total(rowsB)
    libre = _reproduccion_total(rows0)
    if abs(libre - obj) / libre < 1e-6:
        # monocultivo: no hay a quien vetar y la regla parcial es vacua.
        return dict(objetivo=obj, libre=libre, frac=0.0, err=0.0, traza=[],
                    vacua=True)

    def evaluar(frac):
        rows = run_grid([spec(cond="Ex", seed=s, T=T,
                              cfg={**base, "suppress": "open",
                                   "suppress_frac": float(frac)})
                         for s in SEEDS_PILOTO], quiet=True)
        return _reproduccion_total(rows)

    # mas supresion -> menos nacimientos: la relacion es monotona decreciente
    lo, hi = 0.0, 0.9
    traza = []
    mejor, err_mejor = 0.0, 9e9
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        got = evaluar(mid)
        err = abs(got - obj) / obj
        traza.append((mid, got, err))
        if err < err_mejor:
            mejor, err_mejor = mid, err
        if got > obj:          # todavia se reproduce de mas: vetar mas
            lo = mid
        else:
            hi = mid
    return dict(objetivo=obj, libre=libre, frac=mejor, err=err_mejor,
                traza=traza, vacua=False)


def calibrar_proveedor(n0_L1):
    """Iguala el VOLUMEN TOTAL EMITIDO por el proveedor entre B y S.

    Se informan las dos convenciones de igualado, porque la eleccion cambia el
    numero y no hay una respuesta canonica:

      per_capita  cada receptor recibe la misma tasa; el volumen total crece
                  con el numero de receptores
      total       el volumen emitido es constante; la tasa por receptor cae al
                  abrirse la regla

    Informar una sola de las dos sin decir cual es lo que el manuscrito llama
    "un estudio que no informa su igualado".
    """
    base = {"n0_L1": n0_L1, "Q_rate": 0.05}
    rowsB = run_grid([spec(cond="Ex", seed=s, T=T,
                           cfg={**base, "provider": "partial"})
                      for s in SEEDS_PILOTO], quiet=True)
    vol_obj = float(col(rowsB, "provided").mean())

    # convencion 'total': se fija Q_total para emitir el mismo volumen por paso
    q_total = vol_obj / T
    rowsS = run_grid([spec(cond="Ex", seed=s, T=T,
                           cfg={**base, "provider": "open", "match": "total",
                                "Q_total": q_total})
                      for s in SEEDS_PILOTO], quiet=True)
    vol_S = float(col(rowsS, "provided").mean())
    err = abs(vol_S - vol_obj) / vol_obj if vol_obj > 0 else np.nan
    acq = float(col(rowsS, "acq_cost").mean())
    return dict(vol_objetivo=vol_obj, Q_total=q_total, vol_S=vol_S, err=err,
                acq_cost=acq, acq_frac=acq / vol_S if vol_S else np.nan)


# ==========================================================================
# Las tres manipulaciones
# ==========================================================================

def brazos(brazos_def, n0_L1):
    """`brazos_def` es {'B': (cond, cfg), 'R': (...), 'S': (...)}.

    Cada brazo lleva su propia condicion porque el brazo S de la manipulacion 0
    es la MISMA regla de transferencia des-personalizada, es decir `E/x` en
    lugar de `E/p`: mismo volumen, mismo momento, misma topologia, y lo unico
    que cambia es que el dominio deja de incluir la etiqueta.
    """
    base = {"n0_L1": n0_L1}
    specs = []
    for arm, (cond, extra) in brazos_def.items():
        for s in range(N_SEED):
            specs.append(spec(cond=cond, seed=s, T=T, cfg={**base, **extra},
                              arm=arm, label=f"{arm}_{n0_L1}"))
    rows = run_grid(specs, quiet=True)
    out = {arm: [r for r in rows if r["arm"] == arm] for arm in brazos_def}
    return out, rows, specs


def informe(nombre, res, extra_cols=()):
    b = col(res["B"], PHI)
    r = col(res["R"], PHI)
    s = col(res["S"], PHI)
    g = st.resumen_brazos(b, r, s, eps_frac=EPS_FRAC)
    rho_txt = "NA" if g["guardian"] else \
        f"{g['rho'][0]:.3f} [{g['rho'][1]:.3f}, {g['rho'][2]:.3f}]"
    print(f"  {nombre:<26} "
          f"{g['Phi_B'][0]:>9,.0f} {g['Phi_R'][0]:>9,.0f} {g['Phi_S'][0]:>9,.0f}  "
          f"{g['delta'][0]:>+7.3f} [{g['delta'][1]:+.3f},{g['delta'][2]:+.3f}]  "
          f"{rho_txt:>22}")
    if g["guardian"]:
        print(f"  {'':26} guardian: {g['guardian']}")
    return g


def main():
    todo, filas_todas, specs_todas = {}, [], []

    # ----------------------------------------------------------------------
    banner("CALIBRADO DEL IGUALADO  (semillas piloto 1000-1011, se descartan)")
    # ----------------------------------------------------------------------
    cal_sup, cal_prov = {}, {}
    print("  Supresion: se iguala la REPRODUCCION TOTAL realizada en el mundo,")
    print("  que es la consecuencia operativa de impedir la misma cantidad de")
    print("  reproduccion.  Proveedor: se iguala el VOLUMEN EMITIDO.\n")
    print(f"  {'dosis L1':>9} | supresion: {'nac. libre':>11} {'objetivo':>10} "
          f"{'frac S':>7} {'error':>7}"
          f" | proveedor: {'vol B':>11} {'vol S':>11} {'error':>7} {'coste adq':>11}")
    for d in DOSIS:
        cs = calibrar_supresion(d)
        cp = calibrar_proveedor(d)
        cal_sup[d], cal_prov[d] = cs, cp
        marca = " (vacua)" if cs.get("vacua") else ""
        print(f"  {d:>9} | {'':11} {cs['libre']:>11,.0f} {cs['objetivo']:>10,.0f} "
              f"{cs['frac']:>7.4f} {cs['err']:>6.2%}{marca} | {'':4} "
              f"{cp['vol_objetivo']:>11,.0f} {cp['vol_S']:>11,.0f} "
              f"{cp['err']:>6.1%} {cp['acq_cost']:>11,.0f}")
    dentro = all(c["err"] <= TOL for c in cal_sup.values()) and \
             all(c["err"] <= TOL for c in cal_prov.values() if c["vol_objetivo"] > 0)
    print(f"\n  Igualado dentro de la tolerancia pre-registrada (+-{TOL:.0%}): {dentro}")
    if not dentro:
        print("  AVISO: donde el igualado no entra en tolerancia, rho NO esta")
        print("  estimado y asi debe informarse.  Un estudio que no informa su")
        print("  igualado no estima rho.")
    todo["calibrado"] = {"supresion": cal_sup, "proveedor": cal_prov,
                         "dentro_tolerancia": bool(dentro)}

    cab = (f"  {'manipulacion':<26} {'Phi_B':>9} {'Phi_R':>9} {'Phi_S':>9}  "
           f"{'delta':>24}  {'rho':>22}")

    # ----------------------------------------------------------------------
    banner("MANIPULACION 0 (control negativo) — la transferencia en E/p")
    # ----------------------------------------------------------------------
    print("  La auditoria dice que esta regla es PARCIAL (A=1).  La prediccion")
    print("  pre-registrada es que aun asi rho = NA, porque la dependencia basal")
    print("  no es positiva. Es el brazo que comprueba que el metodo no fabrica")
    print("  una medida donde no hay dependencia que fraccionar.\n")
    print(cab)
    m0 = {}
    for d in DOSIS:
        res, rows, sp = brazos({
            "B": ("Ep", {}),                      # transferencia parcial
            "R": ("Ep", {"transfer_on": False}),  # sin transferencia
            "S": ("Ex", {}),                      # misma regla, des-personalizada
        }, n0_L1=d)
        filas_todas += rows
        specs_todas += sp
        m0[d] = informe(f"L1={d}", res)
    todo["manip0_transferencia"] = {str(k): _ser(v) for k, v in m0.items()}

    # ----------------------------------------------------------------------
    banner("MANIPULACION 1 — proveedor de recurso reproductivo")
    # ----------------------------------------------------------------------
    print("  B: el proveedor solo atiende al linaje focal.")
    print("  R: sin proveedor.")
    print("  S: se inscribe quien quiera pagando; volumen emitido igualado.\n")
    print(cab)
    m1 = {}
    for d in DOSIS:
        qt = cal_prov[d]["Q_total"]
        res, rows, sp = brazos({
            "B": ("Ex", {"provider": "partial", "Q_rate": 0.05}),
            "R": ("Ex", {"provider": "none"}),
            "S": ("Ex", {"provider": "open", "match": "total", "Q_total": qt}),
        }, n0_L1=d)
        filas_todas += rows
        specs_todas += sp
        m1[d] = informe(f"L1={d}", res)
        m1[d]["acq_frac"] = cal_prov[d]["acq_frac"]
    todo["manip1_proveedor"] = {str(k): _ser(v) for k, v in m1.items()}
    print("\n  El coste de adquisicion del brazo S se informa aparte y nunca")
    print("  se absorbe en Phi:")
    for d in DOSIS:
        print(f"    L1={d:<4} coste de adquisicion / volumen recibido = "
              f"{cal_prov[d]['acq_frac']:.3f}")

    # ----------------------------------------------------------------------
    banner("MANIPULACION 2 — supresion de competidores")
    # ----------------------------------------------------------------------
    print("  B: se veta la reproduccion de quien no lleva la etiqueta focal.")
    print("  R: sin veto.")
    print("  S: se veta por una condicion de ESTADO simetrica y alcanzable.\n")
    print("  AQUI NO SE CALIBRA UN PUNTO, SE BARRE LA INTENSIDAD.  El calibrado")
    print("  de arriba converge, pero sobre una cantidad que resulta ser casi")
    print("  insensible a la manipulacion: el mundo compensa la reproduccion")
    print("  impedida y los nacimientos totales apenas se mueven en todo el")
    print("  rango de `suppress_frac`.  Igualar sobre una cantidad insensible no")
    print("  es igualar: fracciones muy distintas 'casan' igual de bien y la")
    print("  eleccion entre ellas seria arbitraria.")
    print("  La salida honesta es informar rho en TODO el rango del sustituto.")
    print("  Si no depende de el, la conclusion es robusta a una eleccion de")
    print("  igualado que no sabemos hacer de forma unica.\n")
    print(f"  {'dosis':<8} {'frac S':>7} {'Phi_B':>9} {'Phi_R':>9} {'Phi_S':>9} "
          f"{'delta':>8} {'rho':>22}")
    m2 = {}
    for d in DOSIS:
        base_bs, rows_bs, sp_bs = brazos({
            "B": ("Ex", {"suppress": "partial"}),
            "R": ("Ex", {"suppress": "none"}),
        }, n0_L1=d)
        filas_todas += rows_bs
        specs_todas += sp_bs
        b = col(base_bs["B"], PHI)
        r = col(base_bs["R"], PHI)
        por_frac = {}
        for fr in FRACS_S:
            resS, rowsS, spS = brazos(
                {"S": ("Ex", {"suppress": "open", "suppress_frac": fr})}, n0_L1=d)
            filas_todas += rowsS
            specs_todas += spS
            s = col(resS["S"], PHI)
            g = st.resumen_brazos(b, r, s, eps_frac=EPS_FRAC)
            por_frac[fr] = g
            rho_txt = "NA" if g["guardian"] else \
                f"{g['rho'][0]:.3f} [{g['rho'][1]:.3f}, {g['rho'][2]:.3f}]"
            print(f"  L1={d:<5} {fr:>7.2f} {g['Phi_B'][0]:>9,.0f} "
                  f"{g['Phi_R'][0]:>9,.0f} {g['Phi_S'][0]:>9,.0f} "
                  f"{g['delta'][0]:>+8.3f} {rho_txt:>22}")
        definidos = [g["rho"][0] for g in por_frac.values() if not g["guardian"]]
        if definidos:
            print(f"  {'':8} -> rho en [{min(definidos):.3f}, {max(definidos):.3f}] "
                  f"sobre todo el rango del sustituto")
        # se guarda el punto mediano como representativo, y el rango completo
        medio = FRACS_S[len(FRACS_S) // 2]
        m2[d] = dict(por_frac[medio])
        m2[d]["rango_rho"] = [min(definidos), max(definidos)] if definidos else None
        m2[d]["frac_representativa"] = medio
        print()
    todo["manip2_supresion"] = {str(k): _ser(v) for k, v in m2.items()}

    # ----------------------------------------------------------------------
    banner("RESUMEN: rho como dosis-respuesta")
    # ----------------------------------------------------------------------
    print(f"  {'L1 inicial':>11} {'manip 0 (control)':>20} {'manip 1 (recurso)':>20} "
          f"{'manip 2 (exencion)':>24}")
    for d in DOSIS:
        def fmt(m):
            return "NA" if m[d]["guardian"] else f"{m[d]['rho'][0]:.3f}"
        r2 = m2[d].get("rango_rho")
        t2 = f"[{r2[0]:.2f}, {r2[1]:.2f}]" if r2 else "NA"
        print(f"  {d:>11} {fmt(m0):>20} {fmt(m1):>20} {t2:>24}")
    print("\n  La manipulacion 2 se informa como RANGO sobre todas las")
    print("  intensidades del sustituto, no como punto: ver arriba.")
    print("\n  En L1=0 la regla parcial no tiene a quien desfavorecer: la")
    print("  dependencia basal se anula y rho no se define.  Es H4, y es la")
    print("  demostracion empirica de que la parcialidad de una regla no es")
    print("  la dependencia de un sistema respecto a ella.")

    save("brazos", filas_todas, specs_todas,
         extra={"resumen": todo, "T": T, "n_seed": N_SEED, "dosis": list(DOSIS),
                "tolerancia_igualado": TOL, "eps_frac": EPS_FRAC,
                "seeds_piloto": [SEEDS_PILOTO.start, SEEDS_PILOTO.stop]})
    return 0


def _ser(g):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items()}


if __name__ == "__main__":
    sys.exit(main())
