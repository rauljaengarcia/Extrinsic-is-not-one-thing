#!/usr/bin/env python3
r"""
R1.Q1 (Revisor 1): ¿es robusto el signo de delta a la forma del coste?

La §6.5 y la Tabla 5 dicen que retirar la transferencia operada desde fuera
SUBE los ciclos autonomos: delta = -0.125 a n_L1 = 0, con el coste de
mantenimiento c0 + k_c a^2, k_c = 0.25 y c0 = 0.18.  El revisor pregunta si el
signo se sostiene en un entorno pequeno de (k_c, c0).

PROTOCOLO, identico al de delta a n_L1 = 0: condicion E en monocultivo (E/x; a
n_L1 = 0 es bit a bit E/p, H4), brazo B con la transferencia y brazo R sin ella
(`remove_operator=True`), Phi = phi_focal, T = 20 000, semillas 0..49
compartidas por los dos brazos.  Rejilla 5x5, +-40 % alrededor de (0.25, 0.18):
k_c en {0.15, 0.20, 0.25, 0.30, 0.35} y c0 en {0.10, 0.14, 0.18, 0.22, 0.26}; el
resto de parametros, por defecto.

POR CELDA: delta con intervalo percentil emparejado (10 000 replicas, semilla
BOOT_SEED + 3, la de delta en boot_emparejado.py); su signo (negativo, positivo
o compatible con 0); poblacion final media, afinidad media y numero de
transferencias.  Degeneracion, con el criterio de validacion.py (v5):
MIN_EVENTOS = 500 asignaciones en una ventana de 600 pasos.  Aqui no hay ventana
de acceso, asi que se aplica a la tasa media: transferencias del brazo B / T x
600.  Una celda en la que se extingue alguna semilla se marca como no viable.

REGRESION OBLIGATORIA, antes que nada: en (0.25, 0.18) los Phi por semilla de B
y R tienen que coincidir bit a bit con las filas E/x de
results/ortogonalidad.json para las semillas 0..49 (se compara la fila entera).
Si no coinciden, el protocolo no es el publicado: el script lo dice, guarda lo
corrido y para sin correr el resto de la rejilla.

Se anaden a la tabla las 10 variantes de la rejilla de robustez de
validacion.py (v5), que son evidencia ya publicada sobre theta, k_s, E_don, q y
m: alli Phi_R/Phi_B > 1 equivale a delta < 0.  Miden `births` con 8 semillas y
T = 15 000, y solo hay cociente de medias, sin intervalo.  validacion.json
guarda solo el veredicto de v5; los cocientes se leen de validacion.log, la
salida de esa misma corrida.

    python experiments/r1_sensibilidad_coste.py            # ~45 min con 12 nucleos
    python experiments/r1_sensibilidad_coste.py --solo-figura

Escribe results/r1_sensibilidad_coste.json y figures/r1_sensibilidad.png.
R1_T y R1_SEMILLAS reducen la escala para una prueba de humo, que escribe
r1_sensibilidad_coste_humo.json y no hace la regresion.
"""
import argparse
import os
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "experiments"))

import runner                                          # noqa: E402  (threading)
import numpy as np                                     # noqa: E402
import r1_comun as rc                                  # noqa: E402
import estadistica as st                               # noqa: E402
import estadistica_r1 as er                            # noqa: E402
from runner import spec, run_grid, col, banner         # noqa: E402

KC = (0.15, 0.20, 0.25, 0.30, 0.35)
C0 = (0.10, 0.14, 0.18, 0.22, 0.26)
PUBLICADO = (0.25, 0.18)
T = int(os.environ.get("R1_T", 20_000))
N_SEM = int(os.environ.get("R1_SEMILLAS", 50))
HUMO = (T, N_SEM) != (20_000, 50)
NOMBRE = "r1_sensibilidad_coste" + ("_humo" if HUMO else "")
PHI = "phi_focal"
#: criterio de degeneracion de validacion.py (v5)
MIN_EVENTOS = 500
VENTANA = 600


def etiqueta(kc, c0, arm):
    return f"kc={kc:.2f}|c0={c0:.2f}|{arm}"


def specs_celda(kc, c0):
    return [spec(cond="Ex", seed=s, T=T, cfg={"kc": kc, "c0": c0},
                 remove_operator=(arm == "R"), arm=arm, label=etiqueta(kc, c0, arm))
            for arm in ("B", "R") for s in range(N_SEM)]


def por_semilla(rows, kc, c0, arm):
    sel = [r for r in rows if r["label"] == etiqueta(kc, c0, arm)]
    return sorted(sel, key=lambda r: r["seed"])


# --------------------------------------------------------------------------
def regresion(rows):
    """Cada fila de (0.25, 0.18) frente a su fila E/x publicada."""
    orto = rc.cargar("ortogonalidad")["filas"]
    pub = {(r["arm"], r["seed"]): r for r in orto if r["label"] in ("Ex_B", "Ex_R")}
    difieren, phi_igual = [], 0
    for arm in ("B", "R"):
        for r in por_semilla(rows, *PUBLICADO, arm):
            p = pub[(arm, r["seed"])]
            phi_igual += int(r[PHI] == p[PHI] and r["births"] == p["births"])
            claves = rc.misma_fila(r, p)
            if claves:
                difieren.append({"arm": arm, "seed": r["seed"], "claves": claves,
                                 "phi": r[PHI], "phi_publicado": p[PHI]})
    n = 2 * N_SEM
    return {"filas": n, "phi_identicos": phi_igual, "filas_identicas": n - len(difieren),
            "ok": not difieren and phi_igual == n, "difieren": difieren[:20]}


def resumen_celda(rows, kc, c0):
    fb, fr = por_semilla(rows, kc, c0, "B"), por_semilla(rows, kc, c0, "R")
    assert [r["seed"] for r in fb] == [r["seed"] for r in fr] == list(range(N_SEM))
    b, r = col(fb, PHI), col(fr, PHI)
    d = er.percentil_pareado(st.delta, b, r, seed=st.BOOT_SEED + 3)
    transf = float(col(fb, "transfers").mean())
    eventos = transf / T * VENTANA
    ext_b = int(sum(x["extinct"] for x in fb))
    ext_r = int(sum(x["extinct"] for x in fr))
    degenerada = bool(eventos < MIN_EVENTOS)
    viable = ext_b == 0 and ext_r == 0
    if not viable:
        lectura = "no viable"
    elif degenerada:
        lectura = "degenerada"
    else:
        lectura = er.signo(d)
    return {
        "kc": kc, "c0": c0, "publicado": (kc, c0) == PUBLICADO,
        "delta": rc.iv(d), "signo": er.signo(d), "lectura": lectura,
        "Phi_B": float(b.mean()), "Phi_R": float(r.mean()),
        "Phi_R_sobre_Phi_B": float(r.mean() / b.mean()),
        "n_final_B": float(col(fb, "n_final").mean()),
        "n_final_R": float(col(fr, "n_final").mean()),
        "g_a_B": float(np.nanmean(col(fb, "g_a"))),
        "g_a_R": float(np.nanmean(col(fr, "g_a"))),
        "transfers_B": transf, "eventos_por_600_pasos": eventos,
        "degenerada": degenerada, "extinciones_B": ext_b, "extinciones_R": ext_r,
        "viable": viable,
    }


def rejilla_v5():
    """Las variantes de v5, de validacion.log (validacion.json no las guarda)."""
    lineas = (rc.RESULTS / "validacion.log").read_text(encoding="utf-8",
                                                       errors="replace").splitlines()
    i0 = next(i for i, l in enumerate(lineas) if "variante" in l and "Phi_R/Phi_B" in l)
    out = []
    for l in lineas[i0 + 1:]:
        if not l.strip() or l.strip().startswith("["):
            break
        campos = re.split(r"\s{2,}", l.strip())
        nombre, r_e, r_i = campos[0], float(campos[1]), float(campos[2])
        patron = campos[-1]
        if nombre == "por defecto":
            continue
        out.append({
            "variante": nombre, "Phi_R_sobre_Phi_B": r_e, "en_C_I": r_i,
            # delta del cociente de medias: (1 - r) / max(1, r), sin intervalo
            "delta_equivalente": (1.0 - r_e) / max(1.0, r_e),
            "signo_equivalente": "negativo" if r_e > 1 else ("positivo" if r_e < 1 else "cero"),
            "patron_v5": patron, "degenerada": patron not in ("OK", "ROTO"),
        })
    return out


def frontera(celdas):
    """Donde cambia el signo del punto entre celdas vecinas, interpolado."""
    por = {(c["kc"], c["c0"]): c for c in celdas}
    cortes = []
    # vecinos en k_c (misma c0) y en c0 (mismo k_c)
    for j, c0 in enumerate(C0):
        for i in range(len(KC) - 1):
            a, b = por[(KC[i], c0)], por[(KC[i + 1], c0)]
            if np.sign(a["delta"][0]) != np.sign(b["delta"][0]):
                t = a["delta"][0] / (a["delta"][0] - b["delta"][0])
                cortes.append({"eje": "kc", "c0": c0,
                               "kc_cero": KC[i] + t * (KC[i + 1] - KC[i])})
    for i, kc in enumerate(KC):
        for j in range(len(C0) - 1):
            a, b = por[(kc, C0[j])], por[(kc, C0[j + 1])]
            if np.sign(a["delta"][0]) != np.sign(b["delta"][0]):
                t = a["delta"][0] / (a["delta"][0] - b["delta"][0])
                cortes.append({"eje": "c0", "kc": kc,
                               "c0_cero": C0[j] + t * (C0[j + 1] - C0[j])})
    return cortes


# --------------------------------------------------------------------------
def figura(datos):
    """Mapa de signo y magnitud de delta.  Legible en gris: el signo va tambien
    en el rotulo de cada celda y en la trama (sin trama: delta < 0; ///: > 0;
    puntos: compatible con 0), no solo en el color."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    import estilo
    import salida_figuras
    estilo.aplicar()

    celdas = datos["manifiesto"]["resumen"]["celdas"]
    por = {(c["kc"], c["c0"]): c for c in celdas}
    M = np.array([[por[(kc, c0)]["delta"][0] for kc in KC] for c0 in C0])
    lim = max(0.02, float(np.nanmax(np.abs(M))))
    # divergente: dos tonos Okabe-Ito y un gris neutro en el cero
    cmap = LinearSegmentedColormap.from_list(
        "delta", ["#0072B2", "#F2F2F2", "#D55E00"])
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = ax.imshow(M, origin="lower", cmap=cmap, norm=norm, aspect="auto",
                   extent=(-0.5, len(KC) - 0.5, -0.5, len(C0) - 0.5))
    for j, c0 in enumerate(C0):
        for i, kc in enumerate(KC):
            c = por[(kc, c0)]
            trama = {"negativo": "", "positivo": "///", "compatible": "...",
                     "degenerada": "xx", "no viable": "xx"}.get(c["lectura"], "xx")
            if trama:
                ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False,
                                           hatch=trama, edgecolor="#888888", lw=0))
            glifo = {"negativo": "−", "positivo": "+", "compatible": "0"}.get(
                c["lectura"], "×")
            d = c["delta"]
            fondo = abs(d[0]) / lim
            tinta = "white" if fondo > 0.55 else estilo.TINTA

            def num(x):
                return f"{x:+.3f}".replace("-", "−")
            ax.text(i, j + 0.13, f"{glifo}   {num(d[0])}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=tinta)
            ax.text(i, j - 0.20, f"[{num(d[1])}, {num(d[2])}]", ha="center",
                    va="center", fontsize=6.3, color=tinta)
            if c["publicado"]:
                ax.add_patch(plt.Rectangle((i - 0.47, j - 0.47), 0.94, 0.94, fill=False,
                                           edgecolor=estilo.TINTA, lw=2.0))
    ax.set_xticks(range(len(KC)))
    ax.set_xticklabels([f"{k:.2f}" for k in KC])
    ax.set_yticks(range(len(C0)))
    ax.set_yticklabels([f"{c:.2f}" for c in C0])
    ax.set_xlabel(r"$k_c$  (quadratic maintenance-cost coefficient)")
    ax.set_ylabel(r"$c_0$  (basal maintenance cost)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\delta = (\Phi_B-\Phi_R)/\max(\Phi_B,\Phi_R)$")
    cb.outline.set_visible(False)
    n = datos["manifiesto"]["n_seed"]
    t_txt = f"{datos['manifiesto']['T']:,}".replace(",", r"\,")
    ax.set_title(r"Sign of $\delta$ around the published cost $(k_c, c_0) = (0.25, 0.18)$"
                 + "\n" + f"E in monoculture, arms B and R, {n} paired seeds, "
                 + rf"$T = {t_txt}$", loc="left", fontsize=9.5)
    fig.text(0.01, -0.02, "−  interval below 0 (withdrawing the transfer raises "
             "the cycle count)    +  interval above 0    0  interval covers 0    "
             "boxed: published cell", fontsize=7, color=estilo.TINTA_2, ha="left")
    destino = rc.FIGURES / ("r1_sensibilidad" + ("_humo" if HUMO else ""))
    png, pdf = salida_figuras.guardar(fig, destino)
    plt.close(fig)
    print(f"  -> {png.relative_to(RAIZ)} + {pdf.name}")
    return png


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-figura", action="store_true")
    args = ap.parse_args()
    rc.preparar()
    if args.solo_figura:
        figura(rc.cargar(NOMBRE))
        return 0

    banner(f"R1.Q1 -- signo de delta en (k_c, c0)   T={T:,}  semillas 0..{N_SEM - 1}"
           + ("   [PRUEBA DE HUMO]" if HUMO else ""))
    filas, specs = [], []

    # -- 1. la celda publicada, y la regresion ----------------------------
    sp = specs_celda(*PUBLICADO)
    filas += run_grid(sp)
    specs += sp
    if HUMO:
        reg = {"ok": None, "nota": "prueba de humo: no se compara"}
        print("  regresion: omitida en la prueba de humo")
    else:
        reg = regresion(filas)
        print(f"  regresion frente a ortogonalidad.json (E/x, semillas 0..{N_SEM - 1}): "
              f"{reg['phi_identicos']}/{reg['filas']} Phi identicos, "
              f"{reg['filas_identicas']}/{reg['filas']} filas identicas -> "
              f"{'OK' if reg['ok'] else 'NO COINCIDE'}")
        if not reg["ok"]:
            for x in reg["difieren"]:
                print(f"    {x}")
            print("  PARADA: el protocolo no reproduce lo publicado; no se corre el resto.")
            runner.save(NOMBRE, filas, specs, extra={
                "regresion": reg, "parada": True, "T": T, "n_seed": N_SEM,
                "procedencia": rc.procedencia(__file__, ["ortogonalidad.json"])})
            return 1

    # -- 2. el resto de la rejilla, por columnas de k_c --------------------
    for kc in KC:
        sp = [s for c0 in C0 if (kc, c0) != PUBLICADO for s in specs_celda(kc, c0)]
        print(f"  k_c = {kc:.2f}:", end="")
        filas += run_grid(sp)
        specs += sp

    # -- 3. resumen ---------------------------------------------------------
    celdas = [resumen_celda(filas, kc, c0) for c0 in C0 for kc in KC]
    print(f"\n  {'k_c':>5} {'c0':>5} {'delta [IC 95% emparejado]':>30} {'lectura':>11} "
          f"{'N_B':>6} {'N_R':>6} {'a_B':>6} {'transf. B':>10} {'ev/600':>7}")
    for c in celdas:
        marca = "  <- publicado" if c["publicado"] else ""
        print(f"  {c['kc']:>5.2f} {c['c0']:>5.2f} {rc.fmt(c['delta']):>30} "
              f"{c['lectura']:>11} {c['n_final_B']:>6.0f} {c['n_final_R']:>6.0f} "
              f"{c['g_a_B']:>6.3f} {c['transfers_B']:>10,.0f} "
              f"{c['eventos_por_600_pasos']:>7,.0f}{marca}")
    cuenta = {k: sum(c["lectura"] == k for c in celdas)
              for k in ("negativo", "positivo", "compatible", "degenerada", "no viable")}
    cortes = frontera(celdas)
    print(f"\n  lecturas: {cuenta}")
    print(f"  cambios de signo del punto entre celdas vecinas: {len(cortes)}")
    for x in cortes:
        print(f"    {x}")

    v5 = rejilla_v5() if not HUMO else []
    if v5:
        print("\n  rejilla de robustez v5 (validacion.log; births, 8 semillas, T=15 000; "
              "sin intervalo):")
        for x in v5:
            print(f"    {x['variante']:<14} Phi_R/Phi_B = {x['Phi_R_sobre_Phi_B']:.3f}  "
                  f"delta equiv. = {x['delta_equivalente']:+.3f}  "
                  f"({x['signo_equivalente']}{', DEGENERADA en v5' if x['degenerada'] else ''})")

    runner.save(NOMBRE, filas, specs, extra={
        "resumen": {"celdas": celdas, "lecturas": cuenta, "frontera": cortes},
        "regresion": reg, "rejilla_v5": v5,
        "T": T, "n_seed": N_SEM, "kc": list(KC), "c0": list(C0),
        "publicado": list(PUBLICADO), "phi": PHI,
        "bootstrap": {"tipo": "percentil emparejado por semilla", "n_boot": st.N_BOOT,
                      "semilla": st.BOOT_SEED + 3},
        "degeneracion": {"MIN_EVENTOS": MIN_EVENTOS, "ventana": VENTANA,
                         "regla": "transfers del brazo B / T * 600 < MIN_EVENTOS"},
        "procedencia": rc.procedencia(__file__, ["ortogonalidad.json", "validacion.json",
                                                 "validacion.log"]),
    })
    figura(rc.cargar(NOMBRE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
