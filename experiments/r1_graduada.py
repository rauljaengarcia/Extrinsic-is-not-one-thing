#!/usr/bin/env python3
r"""
R2.1 (Revisor 2): reglas graduadas -- reparto fraccionado (w) y segregado en el
tiempo (tau).  El sustrato es `src/ecologia_graduada.py`.

PROTOCOLO de reetiquetado IDENTICO al de exp_ortogonalidad.py: T = 20 000, 150
clones inyectados en t = 10 000, ventana de acceso [10 000, 10 600), semillas
0..99.  Se gradua la regla parcial (cond Ip y Ep):

    w   in {0, 0.25, 0.5, 0.75, 1}   admision fraccionada
    tau in {0.25, 0.5, 0.75}         segregacion temporal, PERIODO = 20 pasos

y se mide lo mismo que en r1_topologia.py (kappa, s, flujos 2x2 y razon de
nacimientos), con el mismo bootstrap percentil emparejado.

Comprobaciones a escala completa, sin coste extra porque esos runs se hacen de
todos modos: E con w = 0 == E/p, E con w = 1 == E/x e I con w = 0 == I/p, fila a
fila frente a ortogonalidad.json (100 semillas).  I con w = 1 no es I/x bit a
bit (ver ecologia_graduada.py); se exige que su kappa y su s caigan dentro del
intervalo emparejado de I/x.  Los extremos de tau (0 y 1) no estan en la
rejilla del encargo; su igualdad bit a bit con x y p la comprueba
tests/test_r1.py.

    python experiments/r1_graduada.py               # ~35 min con 12 nucleos
    python experiments/r1_graduada.py --solo-figura

Escribe results/r1_graduada.json y figures/r1_graduada.png.  R1_T y R1_SEMILLAS
reducen la escala para una prueba de humo (r1_graduada_humo.json, sin
comparacion con lo publicado).
"""
import argparse
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "experiments"))

import runner                                          # noqa: E402
import numpy as np                                     # noqa: E402
import r1_comun as rc                                  # noqa: E402
import estadistica as st                               # noqa: E402
import estadistica_r1 as er                            # noqa: E402
from ecologia_graduada import PERIODO, spec_graduada, run_grid_graduada  # noqa: E402
from r1_topologia import medir                         # noqa: E402
from runner import banner                              # noqa: E402

T = int(os.environ.get("R1_T", 20_000))
N_SEM = int(os.environ.get("R1_SEMILLAS", 100))
HUMO = (T, N_SEM) != (20_000, 100)
NOMBRE = "r1_graduada" + ("_humo" if HUMO else "")
T_INY, N_INY, T_VENTANA = T // 2, 150, 600       # como exp_ortogonalidad.py
WS = (0.0, 0.25, 0.5, 0.75, 1.0)
TAUS = (0.25, 0.5, 0.75)
FAMILIAS = {"I": "Ip", "E": "Ep"}
#: claves que solo tiene la salida graduada, o que no son del run
NO_COMPARAR = ("label", "arm", "cond", "w", "tau", "periodo", "ajenos_vistos",
               "ajenos_admitidos", "pasos_parciales", "pasos_abiertos")


def etiqueta(fam, par, v):
    return f"{fam}|{par}={v:g}"


def specs():
    iny = dict(T=T, inject_at=T_INY, n_inject=N_INY, access_from=T_INY,
               access_until=T_INY + T_VENTANA)
    out = []
    for fam, cond in FAMILIAS.items():
        for w in WS:
            out += [spec_graduada(cond=cond, seed=s, w=w, label=etiqueta(fam, "w", w), **iny)
                    for s in range(N_SEM)]
        for tau in TAUS:
            out += [spec_graduada(cond=cond, seed=s, tau=tau, label=etiqueta(fam, "tau", tau),
                                  **iny) for s in range(N_SEM)]
    return out


def comparar_publicado(filas, orto):
    """Extremos de w frente a las filas de reetiquetado de ortogonalidad.json."""
    pub = {}
    for r in orto["filas"]:
        if r["label"].endswith("_iny"):
            pub[(r["label"][:2], r["seed"])] = r
    casos = {("E", 0.0): "Ep", ("E", 1.0): "Ex", ("I", 0.0): "Ip"}
    out = {}
    for (fam, w), cond in casos.items():
        sel = sorted((r for r in filas if r["label"] == etiqueta(fam, "w", w)),
                     key=lambda r: r["seed"])
        malas = [(r["seed"], rc.misma_fila(r, pub[(cond, r["seed"])], NO_COMPARAR))
                 for r in sel]
        malas = [m for m in malas if m[1]]
        out[f"{fam} w={w:g} == {cond}"] = {"filas": len(sel), "identicas": len(sel) - len(malas),
                                          "ok": not malas, "difieren": malas[:10]}
    return out


def resumen(filas):
    out = []
    for fam in FAMILIAS:
        for par, valores in (("w", WS), ("tau", TAUS)):
            for v in valores:
                sel = [r for r in filas if r["label"] == etiqueta(fam, par, v)]
                m = medir(sel)
                vistos = sum(r["ajenos_vistos"] for r in sel)
                adm = sum(r["ajenos_admitidos"] for r in sel)
                pp = sum(r["pasos_parciales"] for r in sel)
                pa = sum(r["pasos_abiertos"] for r in sel)
                m.update(familia=fam, cond=FAMILIAS[fam], parametro=par, valor=v,
                         fraccion_admitida=(adm / vistos) if vistos else None,
                         fraccion_pasos_parciales=(pp / (pp + pa)) if (pp + pa) else None,
                         extinciones=int(sum(r["extinct"] for r in sel)))
                out.append(m)
    return out


def figura(datos, topo):
    """kappa y s frente a w y frente a tau, para I y E, con los tres regimenes de
    r1_topologia.py como puntos de referencia."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import estilo
    import salida_figuras
    estilo.aplicar()

    res = datos["manifiesto"]["resumen"]["celdas"]
    ref = {m["cond"]: m for m in topo["filas"]}
    estilo_fam = {"I": dict(color=estilo.COLOR["Ip"], marker="o", ls="-"),
                  "E": dict(color=estilo.COLOR["Ep"], marker="s", ls="--")}
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.4), sharey="row",
                            gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.08,
                                         "hspace": 0.38})
    for fila, med, nombre in ((0, "kappa", r"$\kappa$  (allocation by resident donors,"
                                           "\nrelabelled / resident, per capita)"),
                              (1, "s", r"$s$  (total supply, relabelled / resident,"
                                       "\nper capita)")):
        for colu, par in ((0, "w"), (1, "tau")):
            ax = axs[fila, colu]
            for fam in ("I", "E"):
                sel = sorted((m for m in res if m["familia"] == fam and m["parametro"] == par),
                             key=lambda m: m["valor"])
                x = np.array([m["valor"] for m in sel])
                y = np.array([m[med][0] for m in sel])
                lo = np.array([m[med][1] for m in sel])
                hi = np.array([m[med][2] for m in sel])
                e = estilo_fam[fam]
                ax.plot(x, y, color=e["color"], ls=e["ls"], lw=1.6, zorder=3)
                ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt=e["marker"], color=e["color"],
                            ms=6, mfc=e["color"], mec="white", mew=1.0, elinewidth=1.1,
                            capsize=2.5, zorder=4, label=f"{fam}: graded {fam}/p")
            # los tres regimenes de r1_topologia: x en w=1 / tau=0; p en w=0 / tau=1.
            # Rotulos hacia fuera de los datos: el punto en 0 a la izquierda, el
            # punto en 1 a la derecha.
            x_x, x_p = (1.0, 0.0) if par == "w" else (0.0, 1.0)
            diamante = dict(marker="D", ms=8, mfc="white", mec=estilo.TINTA, mew=1.3,
                            ls="none", zorder=5)

            def rotulo(xv, yv, txt):
                fuera = -9 if xv == 0.0 else 9
                ax.annotate(txt, (xv, yv), xytext=(fuera, 0), textcoords="offset points",
                            ha="right" if xv == 0.0 else "left", va="center",
                            fontsize=7.5, color=estilo.TINTA, zorder=6,
                            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
            rx = ref["Ix"][med][0]
            ax.plot([x_x], [rx], **diamante)
            rotulo(x_x, rx, "x")
            rip, rep = ref["Ip"][med][0], ref["Ep"][med][0]
            ax.plot([x_p, x_p], [rip, rep], **diamante)
            if rip == rep:
                rotulo(x_p, rip, "I/p, E/p")
            else:
                rotulo(x_p, rip, "I/p")
                rotulo(x_p, rep, "E/p")
            ax.axhline(1.0, color=estilo.TINTA_2, lw=0.8, ls=":", zorder=1)
            ax.axhline(0.0, color=estilo.TINTA_2, lw=0.8, zorder=1)
            ax.set_xlim(-0.36, 1.36)
            ax.set_ylim(-0.08, 1.15)
            if par == "w":
                ax.set_xticks(WS)
                ax.set_xlabel(r"$w$  (probability of admitting a non-own candidate)")
            else:
                ax.set_xticks((0.0,) + TAUS + (1.0,))
                ax.set_xlabel(r"$\tau$  (fraction of steps under the partial rule)")
            if colu == 0:
                ax.set_ylabel(nombre)
            estilo.limpiar(ax)
    axs[0, 0].legend(loc="upper left", frameon=False)
    axs[0, 0].set_title("(a) fractional admission", loc="left")
    axs[0, 1].set_title("(b) segregation in time", loc="left")
    n = datos["manifiesto"]["n_seed"]
    fig.suptitle(f"Graded recipient-partial rules under relabelling, {n} paired seeds.  "
                 "Open diamonds: the binary rules (x exchangeable; I/p partition; "
                 "E/p exclusion)", fontsize=9.5, y=0.995)
    destino = rc.FIGURES / ("r1_graduada" + ("_humo" if HUMO else ""))
    png, pdf = salida_figuras.guardar(fig, destino)
    plt.close(fig)
    print(f"  -> {png.relative_to(RAIZ)} + {pdf.name}")
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-figura", action="store_true")
    args = ap.parse_args()
    rc.preparar()
    if args.solo_figura:
        figura(rc.cargar(NOMBRE), rc.cargar("r1_topologia"))
        return 0

    banner(f"R2.1 -- reglas graduadas   T={T:,}  semillas 0..{N_SEM - 1}  "
           f"periodo de tau = {PERIODO}" + ("   [PRUEBA DE HUMO]" if HUMO else ""))
    sp = specs()
    filas = []
    for fam, cond in FAMILIAS.items():
        # specs() genera las familias en este mismo orden: filas y specs quedan alineadas
        trozo = [s for s in sp if s.label.startswith(fam + "|")]
        print(f"  {fam} ({cond}):", end="")
        filas += run_grid_graduada(trozo)
    assert [(r["label"], r["seed"]) for r in filas] == [(s.label, s.seed) for s in sp]

    celdas = resumen(filas)
    print(f"\n  {'familia':<8} {'par.':<4} {'valor':>5} {'kappa':>24} {'s':>24} "
          f"{'nac. L1/L0':>24} {'admitidos':>9} {'pasos p':>8}")
    for m in celdas:
        fa = "" if m["fraccion_admitida"] is None else f"{m['fraccion_admitida']:.3f}"
        fp = "" if m["fraccion_pasos_parciales"] is None else f"{m['fraccion_pasos_parciales']:.3f}"
        print(f"  {m['familia']:<8} {m['parametro']:<4} {m['valor']:>5.2f} "
              f"{rc.fmt(m['kappa']):>24} {rc.fmt(m['s']):>24} "
              f"{rc.fmt(m['nacimientos']):>24} {fa:>9} {fp:>8}")

    extra = {"T": T, "n_seed": N_SEM, "t_inyeccion": T_INY, "n_inyeccion": N_INY,
             "ventana": [T_INY, T_INY + T_VENTANA], "ws": list(WS), "taus": list(TAUS),
             "periodo_tau": PERIODO,
             "bootstrap": {"tipo": "percentil emparejado por semilla", "n_boot": st.N_BOOT,
                           "semilla": st.BOOT_SEED}}
    if not HUMO:
        orto = rc.cargar("ortogonalidad")
        comp = comparar_publicado(filas, orto)
        print("\n  extremos frente a ortogonalidad.json (fila entera, 100 semillas):")
        for k, v in comp.items():
            print(f"    {k:<14} {v['identicas']}/{v['filas']} filas identicas -> "
                  f"{'OK' if v['ok'] else 'NO'}")
        topo = rc.cargar("r1_topologia")
        ix = next(m for m in topo["filas"] if m["cond"] == "Ix")
        iw1 = next(m for m in celdas if m["familia"] == "I" and m["parametro"] == "w"
                   and m["valor"] == 1.0)
        dentro = {k: bool(er.cubre(ix[k], iw1[k][0])) for k in ("kappa", "s")}
        print(f"  I con w=1 dentro del intervalo de I/x: kappa {iw1['kappa'][0]:.4f} en "
              f"{rc.fmt(ix['kappa'], 4)} -> {dentro['kappa']};  s {iw1['s'][0]:.4f} en "
              f"{rc.fmt(ix['s'], 4)} -> {dentro['s']}")
        extra.update(extremos_bit_a_bit=comp, I_w1_dentro_de_Ix=dentro,
                     I_w1=dict(kappa=iw1["kappa"], s=iw1["s"]),
                     Ix=dict(kappa=ix["kappa"], s=ix["s"]))
        fuentes = ["ortogonalidad.json", "r1_topologia.json"]
    else:
        fuentes = []
    extra["resumen"] = {"celdas": celdas}
    extra["procedencia"] = rc.procedencia(__file__, fuentes)
    runner.save(NOMBRE, filas, sp, extra=extra)
    if (rc.RESULTS / "r1_topologia.json").is_file():
        figura(rc.cargar(NOMBRE), rc.cargar("r1_topologia"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
