#!/usr/bin/env python3
r"""
Bootstrap EMPAREJADO de los tres brazos, y comparacion con el no emparejado.

    python boot_emparejado.py            # tabla comparativa por dosis
    python boot_emparejado.py --json S   # ademas escribe boot_emparejado.json

POR QUE EXISTE
--------------
`exp_brazos.py` genera los tres brazos con `for s in range(N_SEED)` DENTRO del
bucle de brazos, de modo que B, R y S corren sobre las MISMAS semillas 0..99.
Estan emparejados por numeros aleatorios comunes.

`estadistica.boot_stat` remuestrea cada brazo POR SEPARADO, y su propio
docstring declara la precondicion: "lo correcto cuando los brazos son runs
independientes con semillas distintas".  Aqui no lo son.  Remuestrear por
separado descarta la correlacion positiva que induce el emparejamiento, de
modo que los intervalos de delta y rho salen mas anchos de lo que deberian.
Es un error conservador --no infla la significacion, la reduce-- pero puede
estar sosteniendo un hallazgo por anchura en vez de por dato.

Se corrige calculando, no argumentando, y sale gratis: los Phi por semilla ya
estan en `results/brazos.json`, asi que no hay que reejecutar la simulacion.

QUE HACE DISTINTO
-----------------
Por replica se remuestrea UNA vez el vector de indices de semilla y se aplica
el MISMO a los tres brazos, de forma que cada replica compara los brazos sobre
el mismo conjunto de mundos.  Luego se calcula el contraste dentro de la
replica.  10.000 replicas y percentiles 2,5/97,5, igual que el actual.

QUE NO TOCA
-----------
Nada de `src/`.  `src/*.py` esta resumido en `code_sha256_16`, que figura en el
manifiesto de los seis resultados publicados; cambiarlo los invalidaria todos.
Este modulo vive fuera y solo LEE `brazos.json`.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from estadistica import delta, rho, N_BOOT, ALPHA, BOOT_SEED   # noqa: E402

PHI = "phi_focal"
RESULTADOS = RAIZ / "results"
#: mismas etiquetas que usa exp_brazos.py en su resumen
MANIPULACIONES = {
    "manip0_transferencia": "transferencia (control negativo)",
    "manip1_proveedor": "proveedor de recurso",
    "manip2_supresion": "supresion de competidores",
}


# --------------------------------------------------------------------------
# los dos bootstrap
# --------------------------------------------------------------------------

def boot_separado(fn, *muestras, n_boot=N_BOOT, alpha=ALPHA, seed=BOOT_SEED):
    """Replica exacta de `estadistica.boot_stat`: cada brazo por su cuenta."""
    rng = np.random.default_rng(seed)
    ms = [np.asarray(s, dtype=np.float64) for s in muestras]
    ms = [s[np.isfinite(s)] for s in ms]
    if any(s.size == 0 for s in ms):
        return np.nan, np.nan, np.nan
    vals = np.empty(n_boot)
    for b in range(n_boot):
        draw = [s[rng.integers(0, s.size, s.size)].mean() for s in ms]
        vals[b] = fn(*draw)
    return _percentiles(vals, fn, [s.mean() for s in ms], alpha)


def boot_pareado(fn, *muestras, n_boot=N_BOOT, alpha=ALPHA, seed=BOOT_SEED):
    """UN vector de indices por replica, aplicado a los tres brazos."""
    ms = [np.asarray(s, dtype=np.float64) for s in muestras]
    n = {s.size for s in ms}
    if len(n) != 1:
        raise ValueError(f"los brazos no estan emparejados: tamanos {n}")
    n = n.pop()
    # una semilla solo puede entrar en el remuestreo si esta viva en los tres
    # brazos: emparejar exige que la unidad exista en todos
    ok = np.all([np.isfinite(s) for s in ms], axis=0)
    ms = [s[ok] for s in ms]
    n = int(ok.sum())
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)                 # UNA vez, para los tres
        vals[b] = fn(*[s[idx].mean() for s in ms])
    return _percentiles(vals, fn, [s.mean() for s in ms], alpha)


def _percentiles(vals, fn, medias, alpha):
    punto = fn(*medias)
    fin = vals[np.isfinite(vals)]
    if fin.size == 0:
        return float(punto), np.nan, np.nan
    lo, hi = np.percentile(fin, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(punto), float(lo), float(hi)


# --------------------------------------------------------------------------
# datos
# --------------------------------------------------------------------------

#: Las filas comparten `label` entre manipulaciones (`B_200` cubre las tres),
#: asi que hay que separarlas por lo que la corrida dejo grabado.  Verificado
#: contando: cada (arm, dosis) trae 100 filas por manipulacion, salvo el brazo
#: S de la supresion, que barre cinco intensidades y trae 500.
SELECTOR = {
    "manip0_transferencia": {
        "B": lambda r: r["cond"] == "Ep" and r["transfers"] > 0,
        "R": lambda r: r["cond"] == "Ep" and r["transfers"] == 0,
        "S": lambda r: (r["cond"] == "Ex" and not r["provided"]
                        and not r["suppressed"]),
    },
    "manip1_proveedor": {
        "B": lambda r: r["provided"] > 0,
        "R": lambda r: (r["cond"] == "Ex" and not r["provided"]
                        and not r["suppressed"]),
        "S": lambda r: r["provided"] > 0,
    },
    "manip2_supresion": {
        "B": lambda r: r["suppressed"] > 0,
        "R": lambda r: (r["cond"] == "Ex" and not r["provided"]
                        and not r["suppressed"]),
        "S": lambda r: r["suppressed"] > 0,
    },
}


def phis_por_semilla(filas, clave, dosis):
    """{brazo: array de Phi alineado por semilla}, para una manipulacion."""
    sel_fn = SELECTOR[clave]
    out, crudo = {}, {}
    for arm in ("B", "R", "S"):
        sel = [r for r in filas
               if r.get("arm") == arm and r.get("label") == f"{arm}_{dosis}"
               and sel_fn[arm](r)]
        if not sel:
            return None
        por_semilla = {}
        for r in sel:
            por_semilla.setdefault(r["seed"], []).append(r[PHI])
        out[arm] = por_semilla
        crudo[arm] = len(sel)
        if arm == "S" and clave == "manip2_supresion":
            # El brazo S de la supresion BARRE cinco intensidades por semilla.
            # `exp_brazos.py` guarda como representativo el punto mediano
            # (`FRACS_S[len(FRACS_S)//2]`), no el promedio, asi que aqui se
            # toma tambien el mediano: ordenando las filas de cada semilla por
            # intensidad suprimida y quedandose con la de en medio.  Promediar
            # las cinco daria otro numero y dejaria de replicar lo publicado.
            por_intensidad = {}
            for r in sel:
                por_intensidad.setdefault(r["seed"], []).append(
                    (r["suppressed"], r[PHI]))
            out[arm] = {s: [sorted(v)[len(v) // 2][1]]
                        for s, v in por_intensidad.items()}
    semillas = sorted(set.intersection(*[set(v) for v in out.values()]))
    if not semillas:
        return None
    arr = {arm: np.array([np.mean(out[arm][s]) for s in semillas])
           for arm in out}
    return arr, semillas, crudo


def ic(t, d=3):
    if not np.isfinite(t[0]):
        return "NA"
    return f"{t[0]:>7.{d}f} [{t[1]:>7.{d}f},{t[2]:>7.{d}f}]"


def anchura(t):
    return t[2] - t[1] if np.isfinite(t[1]) and np.isfinite(t[2]) else np.nan


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true",
                    help="escribe results/boot_emparejado.json")
    ap.add_argument("--n-boot", type=int, default=N_BOOT,
                    help=f"replicas de bootstrap (por defecto {N_BOOT})")
    ap.add_argument("--corr", action="store_true",
                    help="imprime la correlacion de Pearson entre brazos")
    args = ap.parse_args()
    n_boot = args.n_boot

    datos = json.loads((RESULTADOS / "brazos.json").read_text(encoding="utf-8"))
    filas = datos["filas"]
    man = datos["manifiesto"]
    dosis = man["dosis"]
    eps_frac = 0.05

    conds = sorted({r["cond"] for r in filas})
    print(f"\n  brazos.json: {len(filas)} filas, dosis {dosis}, "
          f"condiciones {conds}")
    print(f"  {N_BOOT:,} replicas, percentiles "
          f"{100 * ALPHA / 2:.1f}/{100 * (1 - ALPHA / 2):.1f}\n")

    salida = {}
    for clave, nombre in MANIPULACIONES.items():
        res_man = man["resumen"].get(clave)
        if not res_man:
            continue
        print("=" * 100)
        print(f"  {nombre}   [{clave}]")
        print("=" * 100)
        print(f"  {'dosis':>6} {'estad.':>6} {'separado (actual)':>26} "
              f"{'EMPAREJADO':>26} {'anchura':>18}")
        print("  " + "-" * 96)
        salida[clave] = {}
        for d in dosis:
            enc = phis_por_semilla(filas, clave, d)
            if not enc:
                print(f"  {d:>6}   (sin filas emparejables)")
                continue
            arr, semillas, crudo = enc
            b, r, s = arr["B"], arr["R"], arr["S"]
            eps = eps_frac * b.mean()
            # contraste con lo publicado, para que la replica sea auditable
            pub = res_man.get(str(d), {})

            fila = {}
            for est, fn, args_, semilla in (
                    ("delta", delta, (b, r), BOOT_SEED + 3),
                    ("rho", lambda x, y, z: rho(x, y, z, eps), (b, r, s),
                     BOOT_SEED + 4)):
                sep = boot_separado(fn, *args_, seed=semilla, n_boot=n_boot)
                par = boot_pareado(fn, *args_, seed=semilla, n_boot=n_boot)
                a_sep, a_par = anchura(sep), anchura(par)
                red = (1 - a_par / a_sep) * 100 if np.isfinite(a_sep) and a_sep else np.nan
                # el guardian publicado manda: si mordio, rho es NA
                nota = "  (NA publicado)" if (est == "rho" and pub.get("guardian")) else ""
                print(f"  {d:>6} {est:>6} {ic(sep):>26} {ic(par):>26} "
                      f"{a_sep:>7.3f}->{a_par:<7.3f} {red:>4.0f}%{nota}")
                fila[est] = {"separado": list(sep), "emparejado": list(par),
                             "anchura_sep": a_sep, "anchura_par": a_par,
                             "publicado": pub.get(est),
                             "guardian": pub.get("guardian")}
            # Correlacion de Pearson entre brazos: es LO QUE EL EMPAREJAMIENTO
            # EXPLOTA.  Positiva -> el pareado estrecha.  Negativa -> lo
            # correcto es que ensanche, y no seria ruido sino un hecho.
            corr = {}
            for a, bb in (("B", "R"), ("B", "S"), ("R", "S")):
                x, y = arr[a], arr[bb]
                corr[f"{a}-{bb}"] = (float(np.corrcoef(x, y)[0, 1])
                                     if x.std() > 0 and y.std() > 0 else float("nan"))
            fila["corr"] = corr
            if args.corr:
                print(f"  {'':6} {'corr':>6} " + "  ".join(
                    f"{k}={v:+.3f}" for k, v in corr.items()))
            fila["n_semillas"] = len(semillas)
            fila["filas_por_brazo"] = crudo
            salida[clave][str(d)] = fila
        print()

    # --- la replica debe reproducir lo publicado antes de creerse el pareado
    print("=" * 100)
    print("  VALIDACION: el bootstrap SEPARADO debe reproducir brazos.json")
    print("=" * 100)
    if n_boot != N_BOOT:
        print(f"  omitida: lo publicado se calculo con {N_BOOT:,} replicas y "
              f"aqui van {n_boot:,}.")
        print(f"  Con --n-boot distinto de {N_BOOT:,} la comparacion no aplica; "
              f"el bootstrap es\n  reproducible con la semilla fija, no "
              f"invariante al numero de replicas.")
        if args.json:
            print("\n  NO se escribe el JSON: solo se publica la corrida "
                  f"de {N_BOOT:,} replicas.")
        return 0
    malos = 0
    for clave, porc in salida.items():
        for d, fila in porc.items():
            for est in ("delta", "rho"):
                pub = fila[est].get("publicado")
                if not pub or fila[est].get("guardian"):
                    continue
                sep = fila[est]["separado"]
                dif = max(abs(a - b) for a, b in zip(pub[:3], sep))
                # el bootstrap es aleatorio pero con semilla fija: debe salir
                # identico, no parecido
                ok = dif < 1e-9
                if not ok:
                    malos += 1
                    print(f"  [DIFIERE] {clave} L1={d} {est}: "
                          f"publicado {pub[:3]} vs replica {sep}")
    if malos:
        print(f"  {malos} puntos NO reproducidos")
    else:
        print("  OK: la replica coincide con lo publicado en todos los puntos")

    if args.json:
        p = RESULTADOS / "boot_emparejado.json"
        p.write_text(json.dumps(salida, indent=1, default=float),
                     encoding="utf-8")
        print(f"\n  escrito {p}")
    return 1 if malos else 0


if __name__ == "__main__":
    sys.exit(main())
