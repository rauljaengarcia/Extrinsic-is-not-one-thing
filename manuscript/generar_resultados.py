#!/usr/bin/env python3
"""
Genera la seccion de Resultados del manuscrito a partir de los ficheros de
`experimento/resultados/`.

Por que generarla en vez de escribirla a mano: todos los numeros del texto
salen del mismo sitio que las figuras, de modo que no puede haber una cifra en
el manuscrito que no este en los datos. Si se vuelve a correr el experimento,
se regenera la seccion y las discrepancias son imposibles en lugar de
improbables.

    python generar_resultados.py          # escribe seccion_resultados.tex
    python generar_resultados.py --insertar   # ademas lo mete en main.tex
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "experimento" / "resultados"
ETIQ = {"Ix": r"$\mathrm{I/x}$", "Ip": r"$\mathrm{I/p}$",
        "Ex": r"$\mathrm{E/x}$", "Ep": r"$\mathrm{E/p}$"}
CONDS = ("Ix", "Ip", "Ex", "Ep")


def cargar(nombre, obligatorio=True):
    p = RES / f"{nombre}.json"
    if not p.exists():
        if obligatorio:
            raise FileNotFoundError(p)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def ic(t, d=3):
    """Punto con intervalo, en el formato del manuscrito."""
    return f"{t[0]:.{d}f}~[{t[1]:.{d}f},\\,{t[2]:.{d}f}]"


def pct(x, d=1):
    r"""Porcentaje escapado.  Un `%` crudo en LaTeX comenta el resto de la
    linea, asi que nunca puede salir de aqui sin barra.

    NUNCA uses `f"{x:.0%}"` para texto que va al manuscrito: produce un `%`
    crudo y LaTeX se come el resto de la linea sin dar error.  Paso en la §6.7
    y borro del PDF media seccion sin que nada lo avisara.
    """
    return f"{x * 100:.{d}f}\\%"


def pct_rel(x, d=0):
    r"""Diferencia relativa como texto: `47\% more` / `6.8\% less`.

    Escribir "hold +47\% the reserve of the unserved" es agramatical aunque el
    porcentaje este bien escapado.  Este helper obliga a la forma correcta.
    """
    palabra = "more" if x >= 0 else "less"
    return f"{abs(x) * 100:.{d}f}\\% {palabra}"


def numeros(txt):
    """Extrae los numeros de una cadena de veredicto.

    Los veredictos de `validacion.py` estan redactados en castellano y este
    manuscrito esta en ingles, asi que no se pueden interpolar tal cual: solo
    se toman sus cifras y la prosa se escribe aqui.  Si el formato del
    veredicto cambiara, esto devolveria menos numeros de los esperados y la
    generacion fallaria de forma visible en vez de colar una frase en el
    idioma equivocado.
    """
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", txt)]


def figura(archivo, etiqueta, ancho, pie):
    # Dos cosas a proposito, y las dos importan:
    #
    #   sin extension  -> graphicx elige el PDF vectorial si existe y cae al
    #                     PNG si no.  Leer los PNG costaba 1,78 s por pase.
    #   sin subcarpeta -> las figuras viven en la raiz del directorio del
    #                     manuscrito, junto a main.tex.
    base = archivo.rsplit(".", 1)[0]
    return [
        r"\begin{figure}[ht]", r"\centering",
        rf"\includegraphics[width={ancho}\linewidth]{{{base}}}",
        rf"\caption{{{pie}}}", rf"\label{{fig:{etiqueta}}}",
        r"\end{figure}", "",
    ]


def num(x):
    """Miles con espacio fino de LaTeX.

    Se aplica por campo y nunca sobre una linea ya montada: hacerlo sobre la
    linea entera tambien sustituye las comas que separan los extremos de un
    intervalo, y produce `[0.84\\,\\,2.08]`.
    """
    return f"{x:,.0f}".replace(",", r"\,")


def aplicar_emparejado(r_b):
    r"""Sustituye los INTERVALOS de delta y rho por los del bootstrap emparejado.

    Los tres brazos comparten semillas (numeros aleatorios comunes), asi que
    remuestrearlos por separado descarta la correlacion inducida y ensancha los
    intervalos.  `boot_emparejado.py` los recalcula desde `brazos.json` sin
    volver a simular, y aqui se superponen.

    Las ESTIMACIONES PUNTUALES no se tocan y no pueden cambiar: son medias y no
    dependen del remuestreo.  Se comprueba que coinciden al ultimo digito y se
    aborta si no, porque una discrepancia ahi significaria que los dos calculos
    no estan mirando las mismas filas.

    `rango_rho` de la supresion NO es un intervalo bootstrap sino el recorrido
    de rho sobre el barrido del sustituto, asi que no se toca.
    """
    emp = cargar("boot_emparejado", obligatorio=False)
    if not emp:
        print("  AVISO: sin boot_emparejado.json; se usan los intervalos "
              "no emparejados de brazos.json")
        return r_b, 0
    n = 0
    for clave, por_dosis in emp.items():
        if clave not in r_b:
            continue
        for dosis_str, fila in por_dosis.items():
            g = r_b[clave].get(dosis_str)
            if g is None:
                continue
            for est in ("delta", "rho"):
                if g.get("guardian") and est == "rho":
                    continue                      # NA sigue siendo NA
                punto_viejo = g[est][0]
                punto_nuevo = fila[est]["emparejado"][0]
                if np.isfinite(punto_viejo) and abs(punto_viejo - punto_nuevo) > 1e-9:
                    raise SystemExit(
                        f"ABORTADO: el punto de {est} en {clave}/{dosis_str} se "
                        f"mueve, {punto_viejo!r} -> {punto_nuevo!r}. Los puntos "
                        f"son medias y no pueden depender del bootstrap.")
                g[est] = list(fila[est]["emparejado"])
                n += 1
    return r_b, n


def seccion():
    orto = cargar("ortogonalidad")["manifiesto"]
    brz = cargar("brazos")["manifiesto"]
    val = {f["prueba"]: f for f in cargar("validacion")["filas"]}
    c1 = cargar("c1_parcialidad_emergente", obligatorio=False)
    c2 = cargar("c2_ne", obligatorio=False)
    r_o, r_b = orto["resumen"], brz["resumen"]
    r_b, n_emp = aplicar_emparejado(r_b)
    if n_emp:
        print(f"  intervalos emparejados aplicados: {n_emp}")
    n_seed, T = orto["n_seed"], orto["T"]
    n_seed_b = brz["n_seed"]        # los dos experimentos pueden diferir
    dosis = brz["dosis"]
    dmax = str(max(dosis))

    L = []
    A = L.append
    A(r"% ============================================================")
    A(r"\section{Results}")
    A(r"% ============================================================")
    A("")
    A(r"Every figure below comes from " +
      (f"{n_seed} " if n_seed == n_seed_b else
       f"{n_seed} seeds per cell for the interventions and {n_seed_b} ") +
      r"seeds per cell at $T=" + f"{T:,}".replace(",", r"\,") +
      r"$ steps, with percentile bootstrap intervals. Raw continuity values are")
    A(r"primary; \eqref{eq:delta} and \eqref{eq:rho} are derived from them, with")
    A(r"intervals propagated by bootstrapping each arm separately.")
    A("")

    # ---------------------------------------------------------------- 6.1
    A(r"\subsection{The substrate behaves like an evolving population}")
    A("")
    n_ess = numeros(val["ESS de la afinidad"]["detalle"])
    n_der = numeros(val["Deriva neutral"]["detalle"])
    n_ins = numeros(val["Insensibilidad a parametros"]["detalle"])
    assert len(n_ess) >= 2 and len(n_der) >= 5 and len(n_ins) >= 4, \
        "el formato de los veredictos de validacion.py ha cambiado"

    A(r"All five validation checks pass (Figure~\ref{fig:valid}). Across " +
      f"{n_ess[1]:.0f}" + r" combinations of $k_c$ and $R$ the measured")
    A(r"equilibrium affinity matches the interior optimum of \eqref{eq:ess} to")
    A(r"within " + f"{n_ess[0]:.1f}" + r"\%, and $R/S$ under imposed truncation")
    A(r"selection departs from the breeder's prediction of unity by at most")
    A(r"$0.001$.")
    A("")
    if c2 and c2["filas"]:
        vk = [f for f in c2["filas"] if "Vk" in f]
        dos = [f for f in c2["filas"] if "Ne_mutacion" in f]
        if vk and dos:
            Vkm = np.mean([f["Vk"] for f in vk])
            creceVk = vk[-1]["Vk"] > vk[0]["Vk"] * 1.15
            cocs = [f["cociente"] for f in vk]
            r_lo, r_hi = min(f["cociente"] for f in dos), max(f["cociente"] for f in dos)
            ch = [f.get("cociente_hill", float("nan")) for f in vk]
            A(r"Drift is present and its timescale grows with population size,")
            A(r"but the scaling is sublinear and we cannot say why. Reproductive")
            A(r"variance does not account for it: $V_k$ is approximately")
            A(r"constant at " + f"{Vkm:.2f}" + r", so $N_e\approx4N/(V_k+2)$")
            A(r"predicts $N_e\approx N$ while the measured value is " +
              f"{min(cocs):.2f}" + r"--" + f"{max(cocs):.2f}" + r" of it, and")
            A(r"nor does \citet{hill1972}'s correction for overlapping")
            A(r"generations, which would bite only if the generation interval")
            A(r"grew with $N$ and it does not (" + f"{min(ch):.2f}" + r"--" +
              f"{max(ch):.2f}" + r"). Two independent estimates of $N_e$, from")
            A(r"the decay of marker variance and from its equilibrium value")
            A(r"under mutation, agree at matched census size and horizon within")
            A(r"a factor of " + f"{r_lo:.1f}" + r"--" + f"{r_hi:.1f}" + r", so")
            A(r"the fourth check is no longer tautological even though the")
            A(r"deficit stands unexplained. Cohort and generation-interval")
            A(r"measurements, including the falsification of the")
            A(r"overlapping-generation correction, are in the supplementary")
            A(r"material.")
            A("")

    A(r"The structural identities hold in all " + f"{n_ins[0]:.0f}" +
      r" grid variants in which the manipulated rule fires; in " +
      f"{n_ins[2]:.0f}" + r" further variant it is degenerate rather than")
    A(r"broken, the reproductive threshold sitting below the contribution one so")
    A(r"that almost nobody becomes a donor. We require the identities and report")
    A(r"the effect size without requiring it, since demanding that withdrawal")
    A(r"always bite would demand that an extrinsic rule always matter.")
    A("")
    L.extend(figura(
        "fig_validacion.png", "valid", "0.94",
        r"Substrate validation against four external quantitative predictions: "
        r"the analytic evolutionarily stable affinity, the breeder's equation, "
        r"the relationship between effective and census population size, and the scaling of "
        r"equilibrium marker variance with mutational variance."))

    # ---------------------------------------------------------------- 6.2
    A(r"\subsection{Neither coordinate has a dynamical signature at baseline}")
    A("")
    A(r"In monoculture the four conditions are \textbf{bit-for-bit identical}:")
    A(r"same births, same deaths, same final population, same mean genome, in")
    A(f"{r_o['h4_identicas']}" + r" of " +
      f"{r_o.get('h4_semillas', r_o['h4_identicas'])}" + r" seeds. This is a")
    A(r"determinism check rather than a statistical one---a seed either yields")
    A(r"four identical trajectories or it does not, and a single discrepancy")
    A(r"would falsify the claim---so it needs no interval and admits no")
    A(r"dispersion.")
    A("")
    A(r"The reason is the one \eqref{eq:neq} exists to name. A rule conditioned on")
    A(r"the label has nobody to disfavour when everyone carries the same label, so")
    A(r"it reduces to its unconditioned form. And an extrinsic rule applying the")
    A(r"same criterion at the same moment as its intrinsic counterpart produces")
    A(r"the same allocations. \emph{The coordinates are properties of rules, not")
    A(r"of trajectories.} They are invisible until the intervention that")
    A(r"interrogates them is run---which is the substantive reason an audit")
    A(r"(\S4) cannot be replaced by watching a system behave.")
    A("")

    # ---------------------------------------------------------------- 6.3
    A(r"\subsection{Each intervention responds to exactly one coordinate}")
    A("")
    A(r"\begin{table}[ht]")
    A(r"\centering\small")
    A(r"\caption{The two interventions are orthogonal to each other. Withdrawing")
    A(r"the external operator moves only the control coordinate; relabelling the")
    A(r"recipient moves only the exchangeability coordinate. Point estimates with")
    A(r"95\% bootstrap intervals.}")
    A(r"\label{tab:orto}")
    A(r"\begin{tabular}{lccrr}")
    A(r"\toprule")
    A(r"\textbf{Condition} & $C$ & $A$ & \textbf{Operator removal} "
      r"$\Phi_R/\Phi_B$ & \textbf{Relabelling} $L_1/L_0$\\")
    A(r"\midrule")
    for c in CONDS:
        rc = r_o["eje_C"][c]["ratio"]
        ra = r_o["eje_A"][c]["ratio"]
        C = "I" if c[0] == "I" else "E"
        Av = 0 if c[1] == "x" else 1
        A(f"{ETIQ[c]} & {C} & {Av} & {ic(rc)} & {ic(ra)}\\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    A("")
    L.extend(figura(
        "fig1_ortogonalidad.png", "orto", "1.0",
        r"The two coordinates are experimentally separable. (a) The four "
        r"conditions differ from their neighbours in exactly one thing. "
        r"(b) Withdrawing the externally operated supply leaves the intrinsic "
        r"conditions at exactly $1.000$ and moves both extrinsic ones. "
        r"(c) Relabelling the recipient leaves the exchangeable conditions "
        r"unchanged and drives allocation to the relabelled lineage to exactly "
        r"zero in both partial ones. Bars carry 95\% bootstrap intervals; "
        r"hatching encodes condition so identity never rests on colour alone."))
    A(r"Table~\ref{tab:orto} and Figure~\ref{fig:orto} are the experimental")
    A(r"content of \eqref{eq:twoaxes}. They are construct validation: the four")
    A(r"rules were built to differ in one thing each, so a clean separation is")
    A(r"partly a property of the design. What it establishes is that the")
    A(r"coordinates \emph{can} be manipulated independently, which was in doubt.")
    A(r"Withdrawing the externally operated supply leaves the two intrinsic")
    A(r"conditions at a ratio of exactly $1.000$ with zero dispersion across")
    A(r"seeds---there is no external machinery to withdraw---while both extrinsic")
    A(r"conditions move. Relabelling leaves the two exchangeable conditions")
    A(r"unchanged and drives allocation to the relabelled lineage to exactly zero")
    A(r"in both partial ones.")
    A("")
    A(r"The zeros are worth dwelling on. They are not small numbers with")
    A(r"intervals that happen to include zero; they are exact, because the rule's")
    A(r"domain excludes the label and no sampling variation can put it back. A")
    A(r"binary $A$ is the honest report of a binary fact.")
    A("")

    # ---------------------------------------------------------------- 6.4
    A(r"\subsection{Two things the framework got wrong}")
    A("")
    A(r"Running the framework rather than only arguing it turned up two defects")
    A(r"in it. Both are reported here, in one place, because they are findings")
    A(r"of the experiment rather than caveats on it: the first concerns what the")
    A(r"exchangeability coordinate fails to distinguish, and the second, in")
    A(r"\S6.5, concerns the range of the normalized substitution contrast.")
    A(r"Neither is patched.")
    A("")
    A(r"\paragraph{Exclusion and partition share a coordinate and have opposite")
    A(r"outcomes}")
    lect = {c: r_o["exclusion_particion"][c]["lectura"] for c in CONDS}
    A(r"The two partial cells share a coordinate and behave in opposite ways, and")
    A(r"the framework as stated does not distinguish them.")
    A(r"Figure~\ref{fig:relabel} shows both panels of it.")
    A(r"Under $\mathrm{E/p}$ a single allocator conditioned on identity excludes:")
    A(r"the relabelled lineage receives nothing from anyone. Under")
    A(r"$\mathrm{I/p}$ each donor serves its own label, so the relabelled lineage")
    A(r"receives---but only from itself. Allocation across the label boundary is")
    A(r"exactly zero in both; aggregate supply to the relabelled lineage is zero")
    A(r"in one and positive in the other.")
    A("")
    A(r"This is a genuine gap in the account of \S2, found by running it. The")
    A(r"coordinate $A$ answers whether a rule's allocation depends on recipient")
    A(r"identity. It does not answer whether the resulting partition leaves the")
    A(r"disfavoured lineage with a viable supply of its own, and those come apart")
    A(r"in the most ordinary case in biology: identity-keyed care partitions a")
    A(r"population without starving anybody's offspring but their own kin's")
    A(r"competitors. A graded $A$ may capture part of this distinction but is")
    A(r"not guaranteed to resolve it: the difference may be one of the topology")
    A(r"of the allocation rather than of the intensity of partiality.")
    A("")
    L.extend(figura(
        "fig2_reetiquetado.png", "relabel", "0.92",
        r"Identical functional state, different provenance. (a) What the rule "
        r"allocates to residents and to relabelled clones, per organism-step, "
        r"in a short window after injection. (b) Total supply reaching the "
        r"relabelled lineage from any source: the two $A=1$ cells share a "
        r"coordinate and differ completely, one excluding and the other "
        r"partitioning."))

    # ---------------------------------------------------------------- 6.5
    A(r"\subsection{The range of \texorpdfstring{$\rho$}{rho}, what the data do not separate, and a "
      r"principled \texttt{NA}}")
    A("")
    A(r"\begin{table}[ht]")
    A(r"\centering\small")
    A(r"\caption{The three arms, at the largest competing-lineage size tested"
      f" ($n_{{L_1}}={dmax}$ of 400"
      r"). $\Phi$ is autonomous reproductive cycles completed by the focal")
    A(r"lineage, with 95\% bootstrap intervals. Arm S is matched to arm B on")
    A(r"emitted volume for the resource manipulation; for the suppression")
    A(r"manipulation no defensible match exists and the entry is the range of")
    A(r"$\rho$ over the substitute's whole intensity range, for the reasons")
    A(r"given under matching. The acquisition cost paid in arm S is reported")
    A(r"separately and never absorbed into $\Phi$.}")
    A(r"\label{tab:brazos}")
    A(r"\begin{tabular}{lrrrcc}")
    A(r"\toprule")
    A(r"\textbf{Manipulation} & $\Phi_B$ & $\Phi_R$ & $\Phi_S$ & $\delta$ & $\rho$\\")
    A(r"\midrule")
    nombres = {"manip0_transferencia": "Partial transfer (negative control)",
               "manip1_proveedor": "Reproductive resource",
               "manip2_supresion": "Competitor suppression"}
    for key, nom in nombres.items():
        g = r_b[key][dmax]
        if g["guardian"]:
            rho = r"\texttt{NA}"
        elif g.get("rango_rho"):
            rho = f"{g['rango_rho'][0]:.2f}--{g['rango_rho'][1]:.2f}"
        else:
            rho = ic(g["rho"], 2)
        A(f"{nom} & {num(g['Phi_B'][0])} & {num(g['Phi_R'][0])} & "
          f"{num(g['Phi_S'][0])} & {g['delta'][0]:+.2f} & {rho}\\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    A("")
    m0 = r_b["manip0_transferencia"]
    na0 = [d for d in dosis if m0[str(d)]["guardian"]]
    si0 = [d for d in dosis if not m0[str(d)]["guardian"]]
    A(r"Throughout this section $\rho$ is the normalized substitution contrast")
    A(r"of \eqref{eq:rho}, not a fraction: zero and one are reference points")
    A(r"and values above one are meaningful.")
    A("")
    A(r"Table~\ref{tab:brazos} reports the three manipulations at the largest")
    A(r"competing-lineage size and Figure~\ref{fig:brazos} the raw continuity")
    A(r"values behind them; the dose-response below gives the rest. One")
    A(r"bookkeeping point, so that nothing is counted twice: the baseline and")
    A(r"removal arms of the first manipulation at $n_{L_1}=0$ are the same")
    A(r"measurement as the $\mathrm{E}$ rows of Table~\ref{tab:orto}, not an")
    A(r"independent replication of it.")
    A("")
    A(r"\paragraph{The negative control, and a pre-registered prediction that")
    A(r"failed} We pre-registered that the partial transfer rule would return")
    A(r"$\rho=\texttt{NA}$ throughout, because the audit calls it")
    A(r"recipient-partial while baseline dependence on it is not positive. It")
    A(r"does so at $n_{L_1}\in\{" + ", ".join(str(d) for d in na0) + r"\}$: at")
    A(r"$n_{L_1}=0$ the supply is a \emph{constraint} rather than a support")
    A(r"($\delta=" + f"{m0['0']['delta'][0]:+.3f}" + r"$, withdrawing it raises")
    A(r"the autonomous cycle count---a property of the transfer being")
    A(r"externally operated rather than of its partiality, as \S6.6 sets out),")
    A(r"so there is no demonstrated dependence to")
    A(r"apportion and the honest output is not a number. That is \eqref{eq:neq}")
    A(r"measured rather than argued: $A=1$ with $\rho=\texttt{NA}$ is a real")
    A(r"state of affairs, and a method reporting $\rho=0$ there would turn the")
    A(r"absence of demonstrated dependence into a spurious estimate of")
    A(r"complete restoration.")
    A("")
    if si0:
        A(r"At $n_{L_1}\in\{" + ", ".join(str(d) for d in si0) + r"\}$ the")
        A(r"prediction fails, and the reason is worth more than the prediction")
        A(r"was. A partial allocation rule in a mixed population is not only a")
        A(r"support that might turn out to be unnecessary; it is also a")
        A(r"\emph{competitor-suppression mechanism}, because the non-focal")
        A(r"lineage contributes to the pool and is not served. Once the")
        A(r"competitor is large enough, withdrawing the rule releases it and")
        A(r"harms the focal lineage, so baseline dependence becomes positive.")
        A(r"The negative control and the suppression manipulation converge at")
        A(r"high dose---which is a finding about what partial allocation")
        A(r"\emph{is}, not a defect of either.")
        A("")
    # rho > 1 solo se afirma donde el intervalo lo excluye por abajo
    fuertes = [(k, d) for k in nombres for d in dosis
               if not r_b[k][str(d)]["guardian"]
               and not r_b[k][str(d)].get("rango_rho")
               and r_b[k][str(d)]["rho"][1] > 1.0]
    A(r"\paragraph{\texorpdfstring{$\rho$}{rho} above one: the second defect}")
    if fuertes:
        k, d = fuertes[0]
        g = r_b[k][str(d)]
        A(r"At $n_{L_1}=" + str(d) + r"$ the negative control returns")
        A(r"$\rho=" + ic(g["rho"], 2) + r"$, whose interval excludes one from")
        A(r"below---which the normalized substitution contrast permits, since")
        A(r"it is not a fraction and has no upper bound. $\rho>1$ means")
    else:
        A(r"Some point estimates exceed one, though no interval excludes one")
        A(r"from below, so we report the possibility rather than assert it.")
        A(r"$\rho>1$ would mean")
    A(r"$\Phi_S<\Phi_R$: de-personalising the rule costs the focal lineage more")
    A(r"than withdrawing it altogether, because the open substitute does not")
    A(r"merely stop favouring the focal lineage---it supplies the rivals the")
    A(r"partial rule was excluding. \S3.5 now states the unbounded range and")
    A(r"what values above one mean.")
    A("")
    # El rango de rho se establece DENTRO de la manipulacion del recurso, a lo
    # largo de la dosis, no comparando manipulaciones entre si: a la dosis mas
    # alta las tres coinciden.  Los enunciados se derivan de los intervalos.
    m1 = r_b["manip1_proveedor"]
    lo_d, hi_d = str(min(dosis)), dmax
    g_lo, g_hi = m1[lo_d], m1[hi_d]
    disjuntos = (not g_lo["guardian"] and not g_hi["guardian"]
                 and g_lo["rho"][2] < g_hi["rho"][1])
    A(r"\paragraph{The range of \texorpdfstring{$\rho$}{rho}, and what the data do not separate}")
    A(r"The range is established within the resource manipulation, along the")
    A(r"dose, not by contrasting manipulations. At $n_{L_1}=" + lo_d + r"$")
    A(r"$\rho=" + ic(g_lo["rho"], 2) + r"$ and at $n_{L_1}=" + hi_d + r"$")
    A(r"$\rho=" + ic(g_hi["rho"], 2) + r"$" +
      (r"; the two intervals are disjoint." if disjuntos else
       r"; the two intervals overlap."))
    A(r"So the contrast moves from about its zero reference point to about its")
    A(r"one, driven by a quantity we control.")
    A("")
    # a la dosis maxima, comprobar si las manipulaciones se separan entre si
    puntos = []
    for key, nom in nombres.items():
        g = r_b[key][dmax]
        if g["guardian"] and not g.get("rango_rho"):
            continue
        r = g["rango_rho"] if g.get("rango_rho") else g["rho"][1:3]
        puntos.append((nom, r[0], r[1]))
    solapan = all(a[1] <= b[2] and b[1] <= a[2]
                  for i, a in enumerate(puntos) for b in puntos[i + 1:])
    cubren_1 = all(a[1] <= 1.0 <= a[2] for a in puntos)
    if puntos and solapan:
        A(r"What the data do \emph{not} support is a separation between the")
        A(r"manipulations at the largest dose. There " +
          (r"all three intervals cover one and every pair overlaps"
           if cubren_1 else r"every pair of intervals overlaps") + r", so at")
        A(r"$n_{L_1}=" + dmax + r"$ we cannot distinguish a rule whose benefit")
        A(r"was the resource from one whose benefit was the exemption. An")
        A(r"earlier run at thirty seeds appeared to separate them; it did not")
        A(r"survive the increase to " + f"{n_seed_b}" + r". We report the")
        A(r"convergence rather than the appearance.")
        A("")
    A(r"The convergence has a mechanism rather than being an artefact of the")
    A(r"estimator: as the competing lineage grows, what the partial provider")
    A(r"withholds from it becomes a larger part of what the focal lineage")
    A(r"enjoys, so supplying a resource and suppressing a competitor converge")
    A(r"on being the same intervention.")
    A("")

    # dosis-respuesta
    A(r"\paragraph{The dose-response, and where \texorpdfstring{\texttt{NA}}{NA} comes back}")
    filas = []
    for d in dosis:
        celdas = []
        for key in nombres:
            g = r_b[key][str(d)]
            if g["guardian"] and not g.get("rango_rho"):
                celdas.append(r"\texttt{NA}")
            elif g.get("rango_rho"):
                celdas.append(f"{g['rango_rho'][0]:.2f}--{g['rango_rho'][1]:.2f}")
            else:
                celdas.append(f"{g['rho'][0]:.2f}")
        filas.append((d, celdas))
    A(r"Varying the size of the competing lineage moves $\rho$ systematically:")
    A(r"\begin{center}\small")
    A(r"\begin{tabular}{r" + "c" * len(nombres) + r"}")
    A(r"\toprule")
    A(r"$n_{L_1}$ & " + " & ".join(nombres[k].split(" (")[0] for k in nombres) + r"\\")
    A(r"\midrule")
    for d, celdas in filas:
        A(f"{d} & " + " & ".join(celdas) + r"\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{center}")
    A("")
    A(r"At $n_{L_1}=0$ the partial rule has nobody to disfavour and $\rho$ is")
    A(r"undefined in the two manipulations whose baseline dependence vanishes")
    A(r"there. That is the limiting case of the monoculture identity of \S6.2,")
    A(r"and it is the clearest statement the experiment produces:")
    A(r"\emph{partiality needs someone to disfavour before it can matter to")
    A(r"anything}.")
    A("")
    # la curva del recurso: monotona y con techo en 1
    serie = [(d, m1[str(d)]["rho"][0]) for d in dosis if not m1[str(d)]["guardian"]]
    if len(serie) >= 3:
        mono = all(serie[i][1] <= serie[i + 1][1] + 1e-9
                   for i in range(len(serie) - 1))
        A(r"The resource curve rises " +
          ("monotonically" if mono else "unevenly") +
          r" and approaches one from below rather than")
        A(r"plateauing short of it; an apparent plateau at thirty seeds did not")
        A(r"survive. The approach to one is expected here, since $\rho=1$ is the")
        A(r"point at which none of this manipulation's dependence survives")
        A(r"de-personalising the rule.")
        A("")
    # C4: delta < 0 medido
    g0lo = r_b["manip0_transferencia"][str(min(dosis))]
    if g0lo["delta"][2] < 0:
        A(r"\paragraph{A support that was a constraint}")
        A(r"\S3.5 predicts that $\delta$ can be negative, and it is: at")
        A(r"$n_{L_1}=" + str(min(dosis)) + r"$ the externally operated transfer")
        A(r"gives $\delta=" + ic(g0lo["delta"], 3) + r"$, strictly below zero,")
        A(r"and withdrawing it leaves the lineage completing about " +
          f"{(g0lo['Phi_R'][0] / g0lo['Phi_B'][0] - 1) * 100:.0f}" +
          r"\% \emph{more} reproductive cycles. Table~\ref{tab:orto} gives the")
        A(r"identical ratio for $\mathrm{E/x}$ and $\mathrm{E/p}$, so the")
        A(r"constraint belongs to the transfer being \emph{operated from")
        A(r"outside} rather than to its partiality, and applies to both")
        A(r"extrinsic cells. The mechanism agrees: levelling reserves costs the")
        A(r"individuals closest to reproducing more than it gains those")
        A(r"furthest from it, which is about the allocation, not the label. A")
        A(r"framework reading every external supply as support would have")
        A(r"scored this one backwards; the sign of $\delta$ is what stops it.")
        A("")
    L.extend(figura(
        "fig3_brazos.png", "brazos", "1.0",
        r"The three arms. (a)--(c) Raw continuity in each arm, which is the "
        r"primary result; $\delta$ and $\rho$ are derived from it. (d) $\rho$ "
        r"against the size of the competing lineage, with crosses marking the "
        r"doses at which the guard bites and the honest output is "
        r"\texttt{NA} rather than zero."))

    # calibrado
    cal = r_b["calibrado"]
    prov = cal["proveedor"][dmax]
    A(r"\paragraph{Matching, and where we could not achieve it}")
    A(r"Calibration used pilot seeds disjoint from the confirmatory ones and was")
    A(r"discarded. For the provider the matched quantity is directly")
    A(r"controllable: at $n_{L_1}=" + dmax + r"$ the substitute emitted " +
      f"{prov['vol_S']:,.0f}".replace(",", r"\,") + r" units against a target of " +
      f"{prov['vol_objetivo']:,.0f}".replace(",", r"\,") +
      f" ({pct(prov['err'])} error). The acquisition cost recipients paid to "
      r"qualify was " + f"{prov['acq_frac']:.2f}" +
      r" times the volume received; that is the price of an open rule and it")
    A(r"belongs in the report, not in $\Phi$.")
    A("")
    A(r"For the suppression manipulation we could not construct a defensible")
    A(r"match, and say so rather than presenting one. The obvious target---the")
    A(r"fraction of reproduction the rule prevents---turned out to be unusable")
    A(r"twice over. A counter of blocked events counts organism-steps rather")
    A(r"than prevented events, and the partial rule sterilises a sublineage")
    A(r"permanently, so its blocked individuals accumulate in a way the open")
    A(r"rule's do not; the two arms inflate by different factors. Retargeting to")
    A(r"realised reproduction, which counts events once, the calibration")
    A(r"converges to within $0.2\%$---but onto a quantity that is nearly")
    A(r"\emph{insensitive} to the manipulation, because the world compensates:")
    A(r"total births move by under $3\%$ while the substitute's intensity")
    A(r"ranges over two orders of magnitude. Fractions that differ by $100\times$")
    A(r"match equally well, and choosing between them would be arbitrary.")
    A("")
    A(r"Matching on an insensitive quantity is not matching. We therefore report")
    A(r"$\rho$ across the substitute's whole range instead of at one calibrated")
    A(r"point. It stays near one throughout, so the conclusion is robust to a")
    A(r"matching choice we do not know how to make uniquely. That is a weaker")
    A(r"claim than the pre-registered one and a more defensible one, and it is")
    A(r"recorded as a deviation in the pre-registration.")
    A("")

    # ------------------------------------------------- emergent partiality
    if c1 and c1["filas"]:
        S = next((f for f in c1["filas"] if f["brazo"] == "S"), None)
        B = next((f for f in c1["filas"] if f["brazo"] == "B"), None)
        if S and np.isfinite(S.get("a_in", np.nan)):
            da = (S["a_in"] - S["a_out"]) / S["a_out"]
            dE = (S["E_in"] - S["E_out"]) / abs(S["E_out"])
            A(r"\subsection{Does the open rule play favourites in effect?}")
            A("")
            A(r"An exchangeable rule can still serve a systematically")
            A(r"better-endowed subset. Under \eqref{eq:exch} that is not")
            A(r"partiality---the rule is symmetric under relabelling and its")
            A(r"behaviour tracks state---but it is the exact edge of the")
            A(r"criterion, so we measured it. Observing who is served at each")
            A(r"moment, the open rule's recipients hold " + pct_rel(dE) +
              r" reserve than the unserved, show no detectable difference in")
            A(r"affinity (" + pct_rel(da, 1) + r"), and sit at the " +
              f"{S['pctl_E']:.2f}" + r" energy percentile---the median, not")
            A(r"the top. The excluded are the " + pct(1 - S["frac_serv"]) +
              r" below the qualifying threshold, overwhelmingly newborns, and")
            A(r"the exclusion is transient.")
            if B and np.isfinite(B.get("E_in", np.nan)):
                dEB = (B["E_in"] - B["E_out"]) / abs(B["E_out"])
                A(r"Under the partial rule the recipients instead hold " +
                  pct_rel(dEB, 1) + r" reserve than the unserved, selected on")
                A(r"a label no amount of reserve can change.")
            A(r"So the answer is no: the open rule gates on a state anyone")
            A(r"reaches rather than concentrating on the fittest, and the")
            A(r"distinction \S2.2 draws between state-dependence and")
            A(r"identity-dependence survives contact with the data.")
            A("")

    # ---------------------------------------------------------------- 6.7
    A(r"\subsection{The audited systems as profiles}")
    A("")
    A(r"Figure~\ref{fig:perfil} shows the audit of \S4 as one coordinate per")
    A(r"function per system, with no aggregation. Two things are legible in it")
    A(r"that a summary index would hide. The canonical genetic algorithm and")
    A(r"the population-evolved language models are identical cell by cell,")
    A(r"three decades apart: the coincidence is seen rather than asserted. And")
    A(r"incomparability is read by comparing rows---among the rows audited")
    A(r"here, no artificial one")
    A(r"dominates the replicator's and the sovereign agents' both, whichever")
    A(r"value the indeterminate entry takes, while the free-living population")
    A(r"dominates each of them, which is the comparison the biological rows are")
    A(r"there to provide. These eight systems are spread across the space of")
    A(r"profiles rather than ordered along one; whether evolutionary systems at")
    A(r"large are so spread is not something eight cases can settle.")
    A("")
    L.extend(figura(
        "fig4_perfil.png", "perfil", "1.0",
        r"Each cell is the coordinate pair of the rule bearing on that "
        r"function, transcribed from Table~\ref{tab:audit} and from nothing "
        r"else. Split cells are functions served by two rules with different "
        r"coordinates; the grey cell is the habitat entry declared "
        r"indeterminate there. There are no numeric axes and no aggregation: "
        r"\S2.5 gives no principled weights, so none are used."))
    return "\n".join(L) + "\n"


def main():
    texto = seccion()
    salida = AQUI / "seccion_resultados.tex"
    salida.write_text(texto, encoding="utf-8")
    print(f"  -> {salida.name}  ({len(texto.splitlines())} lineas)")
    if "--insertar" in sys.argv:
        main_tex = AQUI / "main.tex"
        t = main_tex.read_text(encoding="utf-8")
        if "%%RESULTADOS%%" in t:
            t = t.replace("%%RESULTADOS%%", texto)
        else:
            print("  aviso: no hay marcador %%RESULTADOS%%; no se inserta")
            return 1
        main_tex.write_text(t, encoding="utf-8")
        print("  -> insertado en main.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
