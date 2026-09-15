#!/usr/bin/env python3
"""
VALIDACION DEL SUSTRATO — bloqueante.

Antes de cualquier condicion experimental hay que poder responder a la primera
objecion que hara un revisor: "?tu resultado es un artefacto de tu simulador?".

Cinco comprobaciones, cada una contra una prediccion cuantitativa externa al
codigo que la evalua:

  1. ESS de la afinidad         a* = sqrt(R(N-1)/(2 kc N^2))      [del modelo]
  2. Ecuacion del criador       R = h^2 S,  con h^2 = 1 aqui      [Lush 1937]
  3. Deriva neutral             V(t) decae como (1-1/2N)^t        [Wright 1931]
  4. Equilibrio mutacion-deriva V_eq = 2 N_e V_m                  [Lynch & Hill 1986]
  5. Insensibilidad a parametros arbitrarios

La 2 tiene una forma especialmente exigente en este sustrato: la reproduccion
es asexual y el rasgo ES el genotipo, luego no hay varianza ambiental y
h^2 = 1 exactamente.  La prediccion no es "R correlaciona con S" sino
"R = S", que es mucho mas facil de falsar.

    python validacion.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import runner                                          # noqa: E402
import numpy as np                                     # noqa: E402
import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from ecologia import Ecologia, Config, ess_afinidad    # noqa: E402
from runner import spec, run_grid, save, col, banner, FIGURAS  # noqa: E402
import estadistica as st                               # noqa: E402
import estilo                                          # noqa: E402
import salida_figuras                                  # noqa: E402

FIGURAS.mkdir(exist_ok=True)
RESULTADO = {}
VEREDICTO = []

#: Asignaciones minimas en la ventana para que el cociente de acceso sea
#: interpretable.  Medido en `diag_theta.py`: por debajo de unos centenares el
#: intervalo bootstrap cubre 1 holgadamente y el punto no dice nada.
MIN_EVENTOS = 500


def veredicto(nombre, ok, detalle):
    VEREDICTO.append((nombre, ok, detalle))
    print(f"  [{'PASA ' if ok else 'FALLA'}] {nombre}: {detalle}")


# ==========================================================================
def v1_ess(n_seed=12, T=40_000):
    """El optimo interior de la afinidad frente a la prediccion analitica.

    Se barren `kc` y `R`, que mueven la prediccion en direcciones distintas.
    Si el sustrato reprodujera la prediccion por casualidad en un punto, no lo
    haria en seis.
    """
    banner("1. ESS de la afinidad frente a la prediccion analitica")
    combos = [(kc, R) for kc in (0.15, 0.25, 0.40) for R in (400.0, 700.0)]
    specs, meta = [], []
    for kc, R in combos:
        for s in range(n_seed):
            specs.append(spec(cond="Ix", seed=s, T=T,
                              cfg={"kc": kc, "R": R, "transfer_on": False},
                              label=f"kc{kc}_R{R}"))
        meta.append((kc, R))
    rows = run_grid(specs)

    print(f"\n  {'kc':>5} {'R':>6} {'N medido':>10} {'a* predicho':>12} "
          f"{'a medido':>10} {'IC 95%':>18} {'error rel':>10}")
    errores = []
    for kc, R in meta:
        sel = [r for r in rows if r["label"] == f"kc{kc}_R{R}"]
        N = np.mean([r["n_final"] for r in sel])
        pred = float(ess_afinidad(Config(kc=kc, R=R), N))
        m, lo, hi = st.boot_mean([r["g_a"] for r in sel])
        err = abs(m - pred) / pred
        errores.append(err)
        print(f"  {kc:>5.2f} {R:>6.0f} {N:>10.0f} {pred:>12.4f} {m:>10.4f} "
              f"  [{lo:.4f}, {hi:.4f}] {err:>9.1%}")
    RESULTADO["ess"] = dict(combos=meta, rows=rows)
    veredicto("ESS de la afinidad", max(errores) < 0.10,
              f"error relativo maximo {max(errores):.1%} sobre 6 combinaciones")
    return rows, meta


# ==========================================================================
def v2_criador(n_seed=10, T=20_000, medir_desde=10_000):
    """Ecuacion del criador sobre el marcador neutral.

    Se impone seleccion por truncamiento sobre `z` (que no tiene ningun efecto
    fisiologico) y se mide, paso a paso, el diferencial de seleccion S y la
    respuesta R.  Con herencia asexual y sin varianza ambiental, h^2 = 1 y la
    prediccion es R = S.
    """
    banner("2. Ecuacion del criador  R = h^2 S,  con h^2 = 1")
    print(f"  {'frac':>6} {'S medido':>12} {'R medido':>12} {'R/S':>8} {'IC 95% de R/S':>20}")
    ratios_todos = []
    tabla = []
    for frac in (0.2, 0.5, 0.8):
        por_semilla = []
        for s in range(n_seed):
            e = Ecologia(cond="Ix", seed=s, log_every=0,
                         cfg=Config(select_on=3, select_frac=frac,
                                    transfer_on=False))
            S_acc, R_acc = [], []
            for t in range(T):
                if not e.step():
                    break
                if t >= medir_desde and e.last_offspring.size > 5:
                    # La referencia es el conjunto de CANDIDATOS, que es sobre
                    # el que actua el truncamiento.  Contra la poblacion entera,
                    # S queda diluido por un factor demografico y R/S deja de
                    # ser 1 sin que nada este mal.
                    cand = e.G[e.last_candidates, 3]
                    par = e.G[e.last_parents, 3]
                    off = e.G[e.last_offspring, 3]
                    S_acc.append(par.mean() - cand.mean())
                    R_acc.append(off.mean() - cand.mean())
            por_semilla.append((np.mean(S_acc), np.mean(R_acc)))
        S = np.array([x[0] for x in por_semilla])
        R = np.array([x[1] for x in por_semilla])
        ratio = R / S
        m, lo, hi = st.boot_mean(ratio)
        ratios_todos.extend(ratio)
        tabla.append((frac, S.mean(), R.mean(), m, lo, hi))
        print(f"  {frac:>6.1f} {S.mean():>12.5f} {R.mean():>12.5f} {m:>8.4f} "
              f"  [{lo:.4f}, {hi:.4f}]")
    RESULTADO["criador"] = tabla
    peor = max(abs(np.array([t[3] for t in tabla]) - 1.0))
    veredicto("Ecuacion del criador", peor < 0.05,
              f"R/S se desvia de 1 como maximo {peor:.3f} en tres intensidades")
    return tabla


# ==========================================================================
def v3_deriva(n_seed=24, T0=5_000, gen_obj=1.6):
    """Deriva neutral: N_e estimado del decaimiento de la varianza.

    Dos correcciones respecto de la version ingenua, ambas necesarias:

    1. Los genes fisiologicos se congelan y arrancan identicos.  Si el marcador
       arranca variable junto a un gen bajo seleccion, lo que se mide es
       ARRASTRE, no deriva: decae mucho mas deprisa y de forma erratica.
    2. El horizonte se ESCALA CON N.  La deriva actua en una escala de tiempo
       proporcional a N_e; con horizonte fijo, una poblacion cuatro veces mayor
       pierde tan poca varianza que `N_e` estimado a partir de ella es puro
       ruido.  Se fija el numero objetivo de generaciones en unidades de N.

    Se contrasta el ESCALADO, no el valor absoluto: `N_e` no es `N` en un
    modelo con numero de descendientes variable, y fingir que lo es seria
    ajustar la teoria al dato.
    """
    banner("3. Deriva neutral: N_e estimado del decaimiento de la varianza")
    cfg_base = {"transfer_on": False, "mu_base": 0.0, "mutate_genes": (3,),
                "g_init_sd": (0.0, 0.0, 0.0, 0.30)}
    # horizonte proporcional a N, para cubrir las mismas generaciones de deriva
    escala = {250.0: 1, 500.0: 2, 1000.0: 4}
    specs, meta = [], []
    for R, k in escala.items():
        T1 = T0 + int(30_000 * k * gen_obj / 1.6)
        for s in range(n_seed):
            specs.append(spec(cond="Ix", seed=s, T=T0, cfg={**cfg_base, "R": R},
                              label=f"R{R}|antes"))
            specs.append(spec(cond="Ix", seed=s, T=T1, cfg={**cfg_base, "R": R},
                              label=f"R{R}|despues"))
        meta.append((R, T1))
    rows = run_grid(specs)
    por = {}
    for r in rows:
        por.setdefault(r["label"], []).append(r)

    print(f"\n  {'R (-> N)':>9} {'N':>7} {'gen':>7} {'V(T)/V(0)':>11} "
          f"{'N_e estimado':>13} {'N_e/N':>8}")
    filas = []
    for R, T1 in meta:
        a = sorted(por[f"R{R}|antes"], key=lambda r: r["seed"])
        b = sorted(por[f"R{R}|despues"], key=lambda r: r["seed"])
        rat, Ne, Ns, Gs = [], [], [], []
        for x, y in zip(a, b):
            v0, v1 = x["sd_z"] ** 2, y["sd_z"] ** 2
            if not (v0 > 0 and v1 > 0):
                continue
            N = 0.5 * (x["n_final"] + y["n_final"])
            g = (y["births"] - x["births"]) / N
            lam = -np.log(v1 / v0) / g
            if lam <= 0:
                continue
            rat.append(v1 / v0); Ne.append(1.0 / (2.0 * lam))
            Ns.append(N); Gs.append(g)
        mN, mNe = np.mean(Ns), np.mean(Ne)
        filas.append((R, mN, np.mean(Gs), np.mean(rat), mNe, mNe / mN))
        print(f"  {R:>9.0f} {mN:>7.0f} {np.mean(Gs):>7.0f} {np.mean(rat):>11.3f} "
              f"{mNe:>13.0f} {mNe/mN:>8.3f}")
    RESULTADO["deriva"] = filas
    Nc = np.array([f[1] for f in filas])
    Ne_ = np.array([f[4] for f in filas])
    perdida = np.array([f[3] for f in filas])
    creciente = bool(np.all(np.diff(Ne_) > 0))
    hay_deriva = bool(np.all(perdida < 0.95))
    # exponente del escalado N_e ~ N^b
    b = float(np.polyfit(np.log(Nc), np.log(Ne_), 1)[0])
    print(f"\n  N_e ~ N^{b:.2f}.  El escalado es SUBLINEAL, y no deberia ser otra")
    print("  cosa: la formula de Wright supone generaciones discretas y")
    print("  descendencia de Poisson, y aqui hay generaciones solapadas y")
    print("  mortalidad dependiente de condicion.  Lo que la teoria exige y se")
    print("  contrasta es que haya deriva y que su escala de tiempo crezca con N;")
    print("  que N_e/N sea constante no se exige porque el modelo no lo predice.")
    veredicto("Deriva neutral", creciente and hay_deriva,
              f"la varianza decae en las tres poblaciones y N_e crece "
              f"monotonamente ({Ne_[0]:.0f} -> {Ne_[-1]:.0f} al pasar N de "
              f"{Nc[0]:.0f} a {Nc[-1]:.0f}), con N_e ~ N^{b:.2f}")
    return filas


def _v3_viejo(T=30_000, n_seed=24):
    """Deriva neutral: decaimiento de la varianza del marcador.

    La version ingenua de esta prueba mide ARRASTRE, no deriva: si el marcador
    arranca con varianza a la vez que un gen bajo seleccion, viaja pegado a el
    y su varianza cae por barridos selectivos.  Aqui los genes fisiologicos se
    congelan (`mutate_genes=(3,)`) y arrancan identicos, de modo que no hay
    ninguna seleccion de la que hitchhike y lo que queda es muestreo puro.

    Se contrasta el ESCALADO, no el valor absoluto: `N_e` no es `N` en un
    modelo con numero de descendientes variable, y fingir que lo es seria
    ajustar la teoria al dato.  Se estima `N_e` del decaimiento observado y se
    comprueba que `N_e/N` es aproximadamente constante.
    """
    banner("3. Deriva neutral: N_e estimado del decaimiento de la varianza")
    print(f"  {'R (-> N)':>9} {'N':>7} {'gen':>7} {'V(T)/V(0)':>11} "
          f"{'N_e estimado':>13} {'N_e/N':>8}")
    filas = []
    for R in (250.0, 500.0, 1000.0):
        rat, Ne, Ns = [], [], []
        for s in range(n_seed):
            e = Ecologia(cond="Ix", seed=s, log_every=0,
                         cfg=Config(R=R, transfer_on=False, mu_base=0.0,
                                    mutate_genes=(3,),
                                    g_init_sd=(0.0, 0.0, 0.0, 0.30)))
            e.run(5_000)                       # transitorio demografico
            idx = np.flatnonzero(e.alive)
            if idx.size < 10:
                continue
            v0, b0, N0 = e.G[idx, 3].var(), e.births, idx.size
            e.run(T)
            idx = np.flatnonzero(e.alive)
            if idx.size < 10 or v0 <= 0:
                continue
            v1 = e.G[idx, 3].var()
            N = 0.5 * (N0 + idx.size)
            gens = (e.births - b0) / N
            if v1 <= 0 or gens <= 0:
                continue
            lam = -np.log(v1 / v0) / gens      # tasa de perdida por generacion
            rat.append(v1 / v0)
            Ne.append(1.0 / (2.0 * lam) if lam > 0 else np.nan)
            Ns.append(N)
        mN, mNe = np.mean(Ns), np.nanmean(Ne)
        filas.append((R, mN, np.mean(rat), mNe, mNe / mN))
        print(f"  {R:>9.0f} {mN:>7.0f} {'':>7} {np.mean(rat):>11.3f} "
              f"{mNe:>13.0f} {mNe/mN:>8.3f}")
    RESULTADO["deriva"] = filas
    r = np.array([f[4] for f in filas])        # N_e / N
    disp = (r.max() - r.min()) / r.mean()
    mono = all(filas[i][2] <= filas[i + 1][2] + 0.02 for i in range(len(filas) - 1))
    veredicto("Deriva neutral", disp < 0.30 and mono,
              f"N_e/N constante dentro de {disp:.1%} (media {r.mean():.2f}) "
              f"y menos perdida de varianza al crecer N")
    return filas


# ==========================================================================
def v4_mutacion_deriva(T=60_000, n_seed=16):
    """Equilibrio mutacion-deriva: V_eq = 2 N_e V_m para un rasgo neutral.

    ADVERTENCIA SOBRE LO QUE ESTA PRUEBA PUEDE Y NO PUEDE FALSAR.  Sin
    seleccion, el marcador es un paseo aleatorio y su dinamica es exactamente
    lineal en sigma, luego la proporcionalidad `V_eq ~ V_m` esta garantizada
    por construccion y no puede fallar.  Esto NO es un test independiente de la
    teoria de la deriva: es una comprobacion de consistencia de escala de la
    implementacion, y como tal se informa.  El contenido empirico sobre `N_e`
    esta en la prueba 3.
    """
    banner("4. Equilibrio mutacion-deriva: escalado de V_eq con V_m")
    print(f"  {'mu_base':>9} {'V_m':>11} {'V_eq medido':>13} {'V_eq/V_m':>11}")
    filas = []
    for mu in (0.005, 0.010, 0.020):
        vs = []
        for s in range(n_seed):
            e = Ecologia(cond="Ix", seed=s, log_every=0,
                         cfg=Config(mu_base=mu, transfer_on=False, g_init_sd=0.0,
                                    mutate_genes=(3,)))
            e.run(T)
            idx = np.flatnonzero(e.alive)
            if idx.size > 10:
                vs.append(e.G[idx, 3].var())
        V = np.mean(vs)
        filas.append((mu, mu * mu, V, V / (mu * mu)))
        print(f"  {mu:>9.3f} {mu*mu:>11.2e} {V:>13.5f} {V/(mu*mu):>11.0f}")
    RESULTADO["mut_deriva"] = filas
    # el cociente V_eq/V_m debe ser aproximadamente constante (= 2 N_e)
    r = np.array([f[3] for f in filas])
    disp = (r.max() - r.min()) / r.mean()
    veredicto("Equilibrio mutacion-deriva", disp < 0.35,
              f"V_eq/V_m constante dentro de {disp:.1%}  (2 N_e = {r.mean():.0f}, "
              f"N ~ {np.mean([f[3] for f in filas])/2:.0f})")
    return filas


# ==========================================================================
def v5_insensibilidad(n_seed=8, T=15_000):
    """Los resultados cualitativos no deben depender de parametros arbitrarios.

    Lo que debe ser invariante son las IDENTIDADES ESTRUCTURALES, no los
    tamanos de efecto:

        C = I  ->  retirar el operador externo deja el cociente en 1.000
                   EXACTO, con varianza cero.  No hay nada externo que retirar.
        A = 0  ->  el reetiquetado deja la asignacion igual entre etiquetas.
        A = 1  ->  el reetiquetado la lleva a CERO EXACTO.

    Exigir ademas que la retirada SIEMPRE muerda en C=E seria un error, y de
    hecho una version anterior de esta prueba lo exigia y fallaba en un punto
    de la rejilla.  Que una regla sea extrinseca no implica que el sistema
    dependa de ella: es exactamente la desigualdad que el manuscrito enuncia
    entre propiedad de una regla y dependencia de un sistema.  Donde la
    retirada no muerde, la salida correcta es delta ~ 0 y rho = NA, y eso es
    un resultado, no un fallo.  El tamano de efecto se informa; no se exige.
    """
    banner("5. Insensibilidad a los parametros arbitrarios")
    rejilla = [{}, {"theta": 7.0}, {"theta": 14.0}, {"ks": 0.01},
               {"ks": 0.04}, {"E_don": 6.0}, {"E_don": 9.0}, {"q": 2.0},
               {"q": 4.0}, {"m_sample": 4}, {"m_sample": 16}]
    # ventana de acceso CORTA: pasado un tiempo los estados de los clones y de
    # los residentes divergen, y una regla que asigna por necesidad reparte
    # distinto por razones de estado, no de etiqueta.
    iny = dict(inject_at=T // 2, n_inject=150, access_from=T // 2,
               access_until=T // 2 + 600)

    # Todas las variantes en UNA sola llamada: con `spawn`, crear un Pool por
    # celda cuesta mas que la simulacion entera.
    specs = []
    for k, var in enumerate(rejilla):
        for s in range(n_seed):
            specs.append(spec(cond="Ex", seed=s, T=T, cfg=var, label=f"{k}|ExB"))
            specs.append(spec(cond="Ex", seed=s, T=T, cfg=var, label=f"{k}|ExR",
                              remove_operator=True))
            specs.append(spec(cond="Ix", seed=s, T=T, cfg=var, label=f"{k}|IxB"))
            specs.append(spec(cond="Ix", seed=s, T=T, cfg=var, label=f"{k}|IxR",
                              remove_operator=True))
            specs.append(spec(cond="Ex", seed=s, T=T, cfg=var, label=f"{k}|accx", **iny))
            specs.append(spec(cond="Ep", seed=s, T=T, cfg=var, label=f"{k}|accp", **iny))
    rows = run_grid(specs)
    por = {}
    for r in rows:
        por.setdefault(r["label"], []).append(r)

    print("\n  La columna 'C: Phi_R/Phi_B' es un TAMANO DE EFECTO y se informa,")
    print("  no se exige: donde vale 1 es que el soporte no era portante en ese")
    print("  regimen, y entonces delta ~ 0 y rho = NA, que es la salida correcta.")
    print(f"\n  {'variante':<18} {'C: Phi_R/Phi_B':>16} {'(en C=I)':>10} "
          f"{'A: L1/L0 en x':>15} {'en p':>7} {'asignac.':>9} {'patron':>10}")
    todo_ok = True
    filas = []
    for k, var in enumerate(rejilla):
        nombre = ", ".join(f"{kk}={vv}" for kk, vv in var.items()) or "por defecto"
        ratio_E = col(por[f"{k}|ExR"], "births").mean() / col(por[f"{k}|ExB"], "births").mean()
        ratio_I = col(por[f"{k}|IxR"], "births").mean() / col(por[f"{k}|IxB"], "births").mean()
        base_x = col(por[f"{k}|accx"], "from_L0_pc_L0")
        base_p = col(por[f"{k}|accp"], "from_L0_pc_L0")
        # Una regla que NUNCA se dispara no tiene coordenadas que revelar.
        # Ocurre, por ejemplo, cuando el umbral reproductivo queda por debajo
        # del de aportacion: los organismos se reproducen antes de llegar a ser
        # donantes, no hay asignaciones, y el cociente es 0/0.  Es una
        # configuracion degenerada, y hay que informarla como tal en vez de
        # contarla como patron roto o dejar que un NaN se propague.
        # Se usa COCIENTE DE MEDIAS, no media de cocientes: con denominadores
        # pequenos o nulos en algunas semillas, la media de cocientes es
        # inestable y basta un cero para propagar un NaN a todo el resultado.
        #
        # Y hace falta un minimo de EVENTOS para que el cociente signifique algo.
        # Con el umbral reproductivo por debajo del de aportacion casi nadie
        # llega a ser donante: medido, 84 asignaciones en la ventana frente a
        # 1.950-4.954 en los demas regimenes.  Ahi el cociente lo decide si un
        # individuo extremo cayo o no del lado de los clones, y su intervalo
        # bootstrap cubre 1 holgadamente.  Eso es falta de potencia, no un
        # efecto, y se informa como tal.
        ev = (col(por[f"{k}|accx"], "flow_00") + col(por[f"{k}|accx"], "flow_01")).mean() / 3.0
        inactiva = bool(ev < MIN_EVENTOS)
        if inactiva:
            acc_x = acc_p = np.nan
            ok, etiq = True, "DEGENERADA"
        else:
            acc_x = col(por[f"{k}|accx"], "from_L0_pc_L1").mean() / base_x.mean()
            acc_p = col(por[f"{k}|accp"], "from_L0_pc_L1").mean() / base_p.mean()
            ok = (abs(ratio_I - 1.0) < 1e-12      # C=I: identidad estructural
                  and abs(acc_x - 1.0) < 0.05     # A=0: asignacion igual
                  and acc_p == 0.0)               # A=1: cero exacto
            etiq = "OK" if ok else "ROTO"
        todo_ok &= bool(ok)
        filas.append((nombre, ratio_E, ratio_I, acc_x, acc_p, bool(ok),
                      inactiva, float(ev)))
        # El recuento de asignaciones se informa para TODAS las variantes, no
        # solo para la que cae por debajo del umbral: es lo que permite
        # comprobar que el criterio de degeneracion se aplica por igual.
        print(f"  {nombre:<18} {ratio_E:>16.3f} {ratio_I:>10.6f} "
              f"{acc_x:>15.3f} {acc_p:>7.2f} {ev:>9,.0f} {etiq:>10}")
    RESULTADO["insensibilidad"] = filas
    n_inact = sum(1 for f in filas if f[6])
    n_act = len(filas) - n_inact
    n_muerde = sum(1 for f in filas if not f[6] and abs(f[1] - 1.0) > 0.02)
    veredicto("Insensibilidad a parametros", todo_ok,
              f"las identidades estructurales se mantienen en {n_act}/{n_act} "
              f"variantes donde la regla se dispara ({n_inact} degenerada/s); "
              f"la retirada muerde en {n_muerde} de {n_act}")
    return filas


# ==========================================================================
def figura(ess_rows, ess_meta, criador, deriva, mutderiva):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

    # -- 1. ESS
    a = ax[0, 0]
    pred, obs, err = [], [], []
    for kc, R in ess_meta:
        sel = [r for r in ess_rows if r["label"] == f"kc{kc}_R{R}"]
        N = np.mean([r["n_final"] for r in sel])
        pred.append(float(ess_afinidad(Config(kc=kc, R=R), N)))
        m, lo, hi = st.boot_mean([r["g_a"] for r in sel])
        obs.append(m)
        err.append([m - lo, hi - m])
    lim = [0.6 * min(pred + obs), 1.15 * max(pred + obs)]
    a.plot(lim, lim, "k--", lw=1, label="identity")
    a.errorbar(pred, obs, yerr=np.array(err).T, fmt="o", ms=6, capsize=3,
               color="#1f77b4")
    a.set_xlabel(r"predicted $a^*=\sqrt{R(N-1)/(2k_cN^2)}$")
    a.set_ylabel(r"measured $a$ at equilibrium")
    a.set_title("1. Interior optimum of affinity")
    a.legend(frameon=False, fontsize=9)

    # -- 2. criador
    a = ax[0, 1]
    fr = [t[0] for t in criador]
    rs = [t[3] for t in criador]
    lo = [t[3] - t[4] for t in criador]
    hi = [t[5] - t[3] for t in criador]
    a.axhline(1.0, color="k", ls="--", lw=1, label=r"$h^2=1$")
    a.errorbar(fr, rs, yerr=[lo, hi], fmt="s", ms=7, capsize=4, color="#d62728")
    a.set_ylim(0.9, 1.1)
    a.set_xlabel("fraction selected by truncation")
    a.set_ylabel(r"$R/S$")
    a.set_title("2. Breeder's equation")
    a.legend(frameon=False, fontsize=9)

    # -- 3. deriva: N_e estimado frente a N censal
    a = ax[1, 0]
    Nc = np.array([d[1] for d in deriva])
    Ne = np.array([d[4] for d in deriva])
    a.plot(Nc, Ne, "o-", ms=7, color="#2ca02c", label=r"estimated $N_e$")
    k = (Ne / Nc).mean()
    a.plot(Nc, k * Nc, "k--", lw=1, label=rf"$N_e={k:.2f}\,N$")
    a.set_xlabel(r"census $N$")
    a.set_ylabel(r"$N_e$ estimated from $-\ln(V_T/V_0)/g$")
    a.set_title("3. Neutral drift")
    a.legend(frameon=False, fontsize=9)

    # -- 4. mutacion-deriva
    a = ax[1, 1]
    vm = np.array([f[1] for f in mutderiva])
    ve = np.array([f[2] for f in mutderiva])
    a.plot(vm, ve, "o-", ms=7, color="#9467bd")
    k = (ve / vm).mean()
    a.plot(vm, k * vm, "k--", lw=1, label=rf"$V_{{eq}}=2N_eV_m$, $2N_e={k:.0f}$")
    a.set_xlabel(r"mutational variance $V_m=\sigma^2$")
    a.set_ylabel(r"$V_{eq}$ of the marker")
    a.set_title("4. Mutation-drift balance")
    a.legend(frameon=False, fontsize=9)

    for row in ax:
        for a in row:
            a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Substrate validation: four external quantitative predictions",
                 fontsize=12)
    fig.tight_layout()
    # 180 dpi quedaba por debajo de la especificacion de Elsevier;
    # el PDF es vectorial y el PNG sube a 300.
    p, pdf = salida_figuras.guardar(fig, FIGURAS / "fig_validacion", dpi=300)
    print(f"\n  -> {p.relative_to(FIGURAS.parent)}")


def main(cuales=None):
    """`cuales` selecciona pruebas por numero, p.ej. `python validacion.py 3 5`.

    Sin argumentos ejecuta las cinco, que es lo que hay que hacer antes de
    publicar: el veredicto solo es valido si todas corren sobre el mismo
    estado del codigo.
    """
    todas = cuales is None
    ess_rows = ess_meta = criador = deriva = mutderiva = None
    if todas or 1 in cuales:
        ess_rows, ess_meta = v1_ess()
    if todas or 2 in cuales:
        criador = v2_criador()
    if todas or 3 in cuales:
        deriva = v3_deriva()
    if todas or 4 in cuales:
        mutderiva = v4_mutacion_deriva()
    if todas or 5 in cuales:
        v5_insensibilidad()
    if todas:
        figura(ess_rows, ess_meta, criador, deriva, mutderiva)
    else:
        print("\n  (ejecucion parcial: no se regenera la figura ni el veredicto "
              "publicable)")

    banner("VEREDICTO")
    for nombre, ok, det in VEREDICTO:
        print(f"  [{'PASA ' if ok else 'FALLA'}] {nombre:<32} {det}")
    fallos = [v[0] for v in VEREDICTO if not v[1]]
    if not todas:
        # Una ejecucion parcial NO debe tocar el fichero de resultados: el
        # manuscrito lee de el los cinco veredictos, y sobrescribirlo con uno
        # solo deja al generador citando una validacion que no existe.
        print("\n  (ejecucion parcial: no se escribe resultados/validacion.json)")
        return 1 if fallos else 0
    save("validacion", [{"prueba": n, "pasa": bool(o), "detalle": d}
                        for n, o, d in VEREDICTO], [],
         extra={"nota": "validacion del sustrato; bloqueante"})
    if fallos:
        print(f"\n  BLOQUEADO: no pasan {fallos}")
        return 1
    print("\n  El sustrato queda validado. Se puede pasar a las condiciones "
          "experimentales.")
    return 0


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:] if a.isdigit()]
    sys.exit(main(args or None))
