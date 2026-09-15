#!/usr/bin/env python3
"""
Figuras del manuscrito.  Lee `resultados/*.json` y no vuelve a simular nada:
el analisis esta separado de la simulacion, de modo que regenerar una figura
no puede alterar un resultado.

    python figuras.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np                                     # noqa: E402
import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.patches import Rectangle               # noqa: E402

from runner import load, col, FIGURAS                  # noqa: E402
import estadistica as st                               # noqa: E402
import estilo                                          # noqa: E402
import salida_figuras                                  # noqa: E402
from estilo import COLOR, TRAMA, MARCA, ETIQ, COLOR_ARM, TRAMA_ARM, ETIQ_ARM, TINTA_2

estilo.aplicar()
FIGURAS.mkdir(exist_ok=True)
CONDS = ("Ix", "Ip", "Ex", "Ep")


def _barra_ic(ax, x, punto, lo, hi, color, trama, ancho=0.62):
    ax.bar(x, punto, ancho, color=color, edgecolor="white", linewidth=0.8,
           hatch=trama, zorder=3)
    ax.errorbar(x, punto, yerr=[[punto - lo], [hi - punto]], fmt="none",
                ecolor="#333333", elinewidth=1.1, capsize=3, zorder=4)


# ==========================================================================
def fig_ortogonalidad():
    """LA figura: cada intervencion responde a exactamente una coordenada."""
    d = load("ortogonalidad")
    res = d["manifiesto"]["resumen"]
    rows = d["filas"]
    fig = plt.figure(figsize=(11.2, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.15], wspace=0.32)

    # -- (a) el espacio de dos coordenadas -------------------------------
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-.5, 1.5); ax.set_ylim(-.5, 1.5)
    for (x, y), c in zip([(0, 1), (1, 1), (0, 0), (1, 0)], ["Ix", "Ip", "Ex", "Ep"]):
        ax.add_patch(Rectangle((x - .46, y - .40), .92, .80, facecolor=COLOR[c],
                               alpha=.13, edgecolor=COLOR[c], lw=1.4,
                               hatch=TRAMA[c], zorder=2))
        ax.text(x, y + .17, ETIQ[c], ha="center", va="center", fontsize=13,
                fontweight="bold", color=COLOR[c], zorder=3)
        ax.text(x, y - .13, ["self-replication", "care directed at\nown lineage",
                             "generic\nscheduler", "breeder favouring\na named lineage"][
                    ["Ix", "Ip", "Ex", "Ep"].index(c)],
                ha="center", va="center", fontsize=7, color="#444444", zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["exchangeable\n$A=0$", "recipient-partial\n$A=1$"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["extrinsic\n$C=E$", "intrinsic\n$C=I$"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(a) the four conditions", loc="left")

    # -- (b) eje C: retirada del operador --------------------------------
    ax = fig.add_subplot(gs[1])
    for i, c in enumerate(CONDS):
        r = res["eje_C"][c]["ratio"]
        _barra_ic(ax, i, r[0], r[1], r[2], COLOR[c], TRAMA[c])
    ax.axhline(1.0, color=TINTA_2, ls="--", lw=1, zorder=2)
    ax.set_xticks(range(4)); ax.set_xticklabels([ETIQ[c] for c in CONDS])
    ax.set_ylabel(r"$\Phi_R / \Phi_B$")
    ax.set_title("(b) withdrawal of the external operator", loc="left")
    estilo.limpiar(ax)
    ax.set_ylim(0.0, 1.42)
    ax.text(-0.45, 1.005, "no effect", fontsize=7, color=TINTA_2,
            va="bottom", ha="left")
    ax.annotate("responds to $C$", xy=(1.5, 1.33), fontsize=8, ha="center",
                color="#444444")
    for i, c in enumerate(CONDS):
        v = res["eje_C"][c]["ratio"][0]
        ax.text(i, 0.045, f"{v:.3f}", ha="center", fontsize=7.5,
                color="white", fontweight="bold", zorder=5)

    # -- (c) eje A: reetiquetado -----------------------------------------
    ax = fig.add_subplot(gs[2])
    for i, c in enumerate(CONDS):
        r = res["eje_A"][c]["ratio"]
        _barra_ic(ax, i, r[0], r[1], r[2], COLOR[c], TRAMA[c])
    ax.axhline(1.0, color=TINTA_2, ls="--", lw=1, zorder=2)
    ax.set_xticks(range(4)); ax.set_xticklabels([ETIQ[c] for c in CONDS])
    ax.set_ylabel("access of $L_1$ / access of $L_0$")
    ax.set_title("(c) relabelling of the recipient", loc="left")
    estilo.limpiar(ax)
    ax.set_ylim(0, 1.42)
    ax.annotate("responds to $A$", xy=(1.5, 1.33), fontsize=8, ha="center",
                color="#444444")
    # Un cero exacto dibuja una barra invisible.  Hay que rotularlo, o el
    # lector no distingue "cero medido" de "dato ausente".
    for i, c in enumerate(CONDS):
        v = res["eje_A"][c]["ratio"][0]
        if v == 0.0:
            ax.annotate("exactly 0", xy=(i, 0), xytext=(i, 0.30),
                        ha="center", fontsize=7.5, color=COLOR[c],
                        fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>", color=COLOR[c], lw=1.2))
        else:
            ax.text(i, 0.045, f"{v:.3f}", ha="center", fontsize=7.5,
                    color="white", fontweight="bold", zorder=5)

    fig.suptitle("The two coordinates are experimentally separable: "
                 "each intervention responds to exactly one",
                 fontsize=11, y=1.02)
    p, pdf = salida_figuras.guardar(fig, FIGURAS / "fig1_ortogonalidad")
    plt.close(fig)
    print(f"  -> {p.name} + {pdf.name}")


# ==========================================================================
def fig_lineage():
    """Mismo estado funcional, distinta procedencia."""
    d = load("ortogonalidad")
    res = d["manifiesto"]["resumen"]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))

    a = ax[0]
    w = 0.36
    for i, c in enumerate(CONDS):
        v0 = res["eje_A"][c]["a_L0"]
        v1 = res["eje_A"][c]["a_L1"]
        a.bar(i - w / 2, v0[0], w, color=COLOR[c], edgecolor="white", lw=.8, zorder=3)
        a.bar(i + w / 2, v1[0], w, color=COLOR[c], edgecolor="white", lw=.8,
              hatch="xxx", alpha=.55, zorder=3)
        a.errorbar([i - w / 2, i + w / 2], [v0[0], v1[0]],
                   yerr=[[v0[0] - v0[1], v1[0] - v1[1]], [v0[2] - v0[0], v1[2] - v1[0]]],
                   fmt="none", ecolor="#333333", elinewidth=1.1, capsize=3, zorder=4)
        if v1[0] == 0:
            a.text(i + w / 2, max(v0[0] * .04, 1e-6), "0", ha="center", va="bottom",
                   fontsize=9, fontweight="bold", color="#333333", zorder=5)
    a.set_xticks(range(4)); a.set_xticklabels([ETIQ[c] for c in CONDS])
    a.set_ylabel("resource allocated by $L_0$ donors\n(per organism-step)")
    a.set_title("(a) what the rule allocates", loc="left")
    from matplotlib.patches import Patch
    a.set_ylim(0, a.get_ylim()[1] * 1.30)
    a.legend(handles=[Patch(facecolor="#888888", label="to $L_0$  (residents)"),
                      Patch(facecolor="#888888", hatch="xxx", alpha=.55,
                            label="to $L_1$  (relabelled clones)")],
             loc="upper center", ncol=2, columnspacing=1.2)
    estilo.limpiar(a)

    b = ax[1]
    tx = res["exclusion_particion"]
    for i, c in enumerate(CONDS):
        v = tx[c]["recv_L1"]
        b.bar(i, v[0], 0.62, color=COLOR[c], edgecolor="white", lw=.8,
              hatch=TRAMA[c], zorder=3)
        b.errorbar(i, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="none",
                   ecolor="#333333", elinewidth=1.1, capsize=3, zorder=4)
        etq = {"sin restriccion": "unrestricted",
               "PARTICION: recibe, solo de si mismo": "partition",
               "EXCLUSION: no recibe de nadie": "exclusion"}[tx[c]["lectura"]]
        b.annotate(etq, xy=(i, v[0]), xytext=(0, 7), textcoords="offset points",
                   ha="center", fontsize=8, color="#333333", zorder=5)
    b.set_xticks(range(4)); b.set_xticklabels([ETIQ[c] for c in CONDS])
    b.set_ylabel("total resource received by $L_1$\n(from any source)")
    b.set_title("(b) the two $A=1$ cells are not alike", loc="left")
    estilo.limpiar(b)

    fig.suptitle("Identical functional state, different provenance: the relabelling "
                 "test inside a controlled system", fontsize=11, y=1.01)
    fig.tight_layout()
    p, pdf = salida_figuras.guardar(fig, FIGURAS / "fig2_reetiquetado")
    plt.close(fig)
    print(f"  -> {p.name} + {pdf.name}")


# ==========================================================================
def fig_brazos():
    d = load("brazos")
    res = d["manifiesto"]["resumen"]
    dosis = d["manifiesto"]["dosis"]
    manips = [("manip0_transferencia", "0  partial transfer"),
              ("manip1_proveedor", "1  reproductive resource"),
              ("manip2_supresion", "2  competitor suppression")]

    fig = plt.figure(figsize=(12.0, 4.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.42)

    dmax = str(max(dosis))
    for k, (key, titulo) in enumerate(manips):
        ax = fig.add_subplot(gs[k])
        g = res[key][dmax]
        for i, arm in enumerate(("B", "R", "S")):
            v = g[f"Phi_{arm}"]
            _barra_ic(ax, i, v[0], v[1], v[2], COLOR_ARM[arm], TRAMA_ARM[arm])
        ax.set_xticks(range(3)); ax.set_xticklabels(["B", "R", "S"])
        ax.set_ylabel(r"$\Phi$  (autonomous cycles of the focal lineage)" if k == 0 else "")
        if g["guardian"]:
            rho = "NA"
        elif g.get("rango_rho"):
            rho = f"{g['rango_rho'][0]:.2f}-{g['rango_rho'][1]:.2f}"
        else:
            rho = f"{g['rho'][0]:.2f}"
        ax.set_title(f"({chr(97+k)}) {titulo}", loc="left", fontsize=9, pad=22)
        ax.annotate(r"$\delta$=" + f"{g['delta'][0]:+.2f}    " + r"$\rho$=" + rho,
                    xy=(0, 1.01), xycoords="axes fraction", fontsize=8.5,
                    color="#333333")
        estilo.limpiar(ax)

    # dosis-respuesta
    ax = fig.add_subplot(gs[3])
    marcas = {"manip0_transferencia": "o", "manip1_proveedor": "s",
              "manip2_supresion": "^"}
    colores = {"manip0_transferencia": "#999999", "manip1_proveedor": "#0072B2",
               "manip2_supresion": "#D55E00"}
    vmax = 1.0
    for key, titulo in manips:
        xs, ys, lo, hi, na = [], [], [], [], []
        for dd in dosis:
            g = res[key][str(dd)]
            if g["guardian"] and not g.get("rango_rho"):
                na.append(dd)
                continue
            if g.get("rango_rho"):
                a, b = g["rango_rho"]
                xs.append(dd); ys.append(0.5 * (a + b))
                lo.append(0.5 * (b - a)); hi.append(0.5 * (b - a))
            else:
                xs.append(dd); ys.append(g["rho"][0]); lo.append(0.0); hi.append(0.0)
            vmax = max(vmax, ys[-1] + hi[-1])
        ax.errorbar(xs, ys, yerr=[lo, hi], marker=marcas[key], color=colores[key],
                    capsize=3, elinewidth=1.1, label=titulo.split("\n")[0], zorder=3)
        for dd in na:
            ax.plot([dd], [-0.07], marker="x", color=colores[key], ms=7,
                    clip_on=False, zorder=3)
    ax.axhline(0, color=TINTA_2, lw=.8, ls=":")
    ax.axhline(1, color=TINTA_2, lw=.8, ls=":")
    top = max(1.25, vmax * 1.12)
    ax.set_ylim(-0.18, top)
    ax.set_xlim(-18, max(dosis) + 18)
    ax.text(-10, -0.155, "x = NA (the guard bites)", fontsize=7,
            color="#444444", va="bottom")
    ax.set_xlabel("initial size of the competing lineage $L_1$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("(d) dose-response", loc="left", pad=22)
    ax.legend(loc="upper left", fontsize=7.5, bbox_to_anchor=(0.01, 0.80))
    estilo.limpiar(ax)

    fig.suptitle(r"The three arms: $\rho$ is a measurement with both extremes "
                 "and a principled NA, not a rhetorical device",
                 fontsize=11, y=1.10)
    p, pdf = salida_figuras.guardar(fig, FIGURAS / "fig3_brazos")
    plt.close(fig)
    print(f"  -> {p.name} + {pdf.name}")


# ==========================================================================
def fig_perfil():
    """Perfil categorico: sistemas auditados x funciones evolutivas.

    Sustituye al plano escalar.  Aquel situaba cada sistema por la FRACCION de
    funciones intrinsecas y exchangeable, con las mixtas valiendo un medio:
    dos decisiones arbitrarias que hacen justo lo que §2.5 prohibe, que es
    combinar las coordenadas en un escalar eligiendo pesos sin fundamento.

    Esta rejilla no agrega nada.  Cada celda muestra su par de coordenadas tal
    como aparece en la tabla de auditoria del manuscrito, y las lecturas que
    interesan se obtienen comparando filas:

      - el AG canonico y los LLM poblacionales dan dos filas identicas celda a
        celda, de modo que la coincidencia SE VE en vez de afirmarse;
      - la incomparabilidad se lee comparando filas, sin comprar ningun peso;
      - que los sistemas se repartan en vez de alinearse tambien.
    """
    # Transcrito literalmente de la Tabla 2 del manuscrito.  Unica fuente.
    FILAS = [
        ("Canonical genetic algorithm",        ["E/x", "E/x", "E/x", "E/p", "E/p", "E/p"]),
        ("Population-evolved LLMs",            ["E/x", "E/x", "E/x", "E/p", "E/p", "E/p"]),
        ("Tierra / Avida world",               ["I/x", "I/x", "I/x", "E/x", "E/x", "E/p"]),
        ("Frontier agent under ARA testing",   ["I/x", "E/p", "E/p", "E/x", "E/x", "E/p"]),
        ("Self-funding on-chain replicator",   ["I/x", "I/x+E/x", "E/x", "E/x", "E/x", "E/x"]),
        ("Sovereign agents on a public chain", ["I/x", "I/x+E/x", "I/x", "E/x+E/p", "E/x", "?"]),
        ("Free-living microbial population",   ["I/x", "E/x", "I/x", "I/x", "E/x", "E/x"]),
        ("Obligate symbiont, host recognition", ["I/x", "E/p", "I/x", "E/p", "E/p", "E/p"]),
    ]
    BIOLOGICOS = 2                    # las dos ultimas filas
    FUNC = ["$G$", "$E$", "$R$", "$M$", "$Q$", "$H$"]
    CLAVE = {"I/x": "Ix", "I/p": "Ip", "E/x": "Ex", "E/p": "Ep"}

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    nf, nc = len(FILAS), len(FUNC)

    for i, (nom, celdas) in enumerate(FILAS):
        y = nf - 1 - i
        for j, celda in enumerate(celdas):
            if celda == "?":
                ax.add_patch(Rectangle((j, y), 1, 1, facecolor="#EEEEEE",
                                       edgecolor="white", lw=2, zorder=2))
                ax.text(j + .5, y + .5, "?", ha="center", va="center",
                        fontsize=12, color="#777777", fontweight="bold", zorder=3)
                continue
            partes = celda.split("+")
            ancho = 1.0 / len(partes)
            for k, pz in enumerate(partes):
                c = CLAVE[pz]
                ax.add_patch(Rectangle((j + k * ancho, y), ancho, 1,
                                       facecolor=COLOR[c], alpha=.85,
                                       edgecolor="white", lw=2,
                                       hatch=TRAMA[c], zorder=2))
            ax.text(j + .5, y + .5, celda.replace("+", " / "), ha="center",
                    va="center", fontsize=7.5 if len(partes) > 1 else 9,
                    color="white", fontweight="bold", zorder=3)

    # separador entre sistemas artificiales y referencias biologicas
    ax.plot([0, nc], [BIOLOGICOS, BIOLOGICOS], color="#333333", lw=1.6, zorder=4)
    ax.text(nc + .08, (nf + BIOLOGICOS) / 2, "artificial", fontsize=8,
            color="#555555", va="center", ha="left", rotation=270)
    ax.text(nc + .08, BIOLOGICOS / 2, "biological", fontsize=8, color="#555555",
            va="center", ha="left", rotation=270)

    ax.set_xlim(0, nc); ax.set_ylim(0, nf)
    ax.set_xticks([j + .5 for j in range(nc)])
    ax.set_xticklabels(FUNC, fontsize=11)
    ax.set_yticks([nf - 1 - i + .5 for i in range(nf)])
    ax.set_yticklabels([n for n, _ in FILAS], fontsize=8.5)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    from matplotlib.patches import Patch
    leyenda = [Patch(facecolor=COLOR[c], alpha=.85, hatch=TRAMA[c],
                     edgecolor="white", label=ETIQ[c]) for c in CONDS]
    leyenda.append(Patch(facecolor="#EEEEEE", edgecolor="white",
                         label="declared indeterminate"))
    ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(.5, -.06),
              ncol=5, fontsize=8.5)
    ax.set_title("Coordinates of the rule bearing on each evolutionary "
                 "function, system by system", fontsize=10, pad=26, loc="left")
    p, pdf = salida_figuras.guardar(fig, FIGURAS / "fig4_perfil")
    plt.close(fig)
    print(f"  -> {p.name} + {pdf.name}")


def main():
    hechas = []
    for nombre, fn in (("ortogonalidad", fig_ortogonalidad),
                       ("reetiquetado", fig_lineage),
                       ("brazos", fig_brazos),
                       ("perfil", fig_perfil)):
        try:
            fn()
            hechas.append(nombre)
        except FileNotFoundError as e:
            print(f"  [saltada] {nombre}: falta {Path(str(e.filename)).name}")
    print(f"\n  {len(hechas)} figuras en {FIGURAS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
