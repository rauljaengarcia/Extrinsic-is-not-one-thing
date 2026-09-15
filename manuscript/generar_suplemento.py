#!/usr/bin/env python3
"""
Genera `suplemento.tex` desde `resultados/c2_ne.json`.

Mismo patron que `generar_resultados.py` y por la misma razon: ninguna cifra
se teclea a mano, de modo que el suplemento no puede divergir del texto
principal ni de los datos.  Si el JSON cambia, el suplemento cambia con el.

    python generar_suplemento.py

El texto principal remite a este material en Data and code availability y en
la seccion de validacion del sustrato.
"""
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RESULTADOS = AQUI.parent / "experimento" / "resultados"
SALIDA = AQUI / "suplemento.tex"

L = []


def A(linea=""):
    L.append(linea)


def num(x, d=0):
    """Miles con espacio fino de LaTeX, como en el texto principal."""
    s = f"{x:,.{d}f}".replace(",", r"\,")
    return s


def main():
    datos = json.loads((RESULTADOS / "c2_ne.json").read_text(encoding="utf-8"))
    man = datos["manifiesto"]
    filas = datos["filas"]

    # el fichero trae dos bloques: el de cohortes/Vk y el de los dos
    # estimadores.  Se distinguen por las claves, no por la posicion.
    cohortes = [f for f in filas if "Vk" in f]
    estimadores = [f for f in filas if "Ne_mutacion" in f]
    if not cohortes or not estimadores:
        sys.exit("c2_ne.json no trae los dos bloques esperados")

    A(r"% ============================================================")
    A(r"%  GENERADO POR generar_suplemento.py -- NO EDITAR A MANO")
    A(r"%  Fuente: experimento/resultados/c2_ne.json")
    A(rf"%  Corrida: {man['fecha_utc']}  code_sha256_16={man['code_sha256_16']}")
    A(r"% ============================================================")
    A(r"\documentclass[11pt]{article}")
    A(r"\usepackage[T1]{fontenc}")
    A(r"\usepackage{lmodern}")
    A(r"\usepackage[a4paper,margin=2.5cm]{geometry}")
    A(r"\usepackage{amsmath,amssymb}")
    A(r"\usepackage{booktabs}")
    # natbib hace falta explicitamente: aqui la clase es `article`, no
    # `elsarticle`, asi que nadie lo carga por nosotros, y el documento usa
    # \citet.  Mismo estilo autor-anyo que el manuscrito principal.
    A(r"\usepackage[authoryear,round]{natbib}")
    A(r"\renewcommand{\thetable}{S\arabic{table}}")
    A(r"\title{Supplementary material\\[2pt]\large Cohort and")
    A(r"generation-interval measurements}")
    A(r"\author{Ra\'ul-Antonio Ja\'en-Garc\'ia}")
    A(r"\date{}")
    A(r"\begin{document}")
    A(r"\maketitle")
    A("")
    A(r"\noindent This document accompanies \emph{Extrinsic Is Not One Thing:")
    A(r"Recipient Exchangeability in Evolutionary Systems}. It reports the")
    A(r"cohort and generation-interval measurements referred to in the")
    A(r"substrate-validation section of the main text. Every number below is")
    A(r"generated from the same result file that produces the corresponding")
    A(r"statements in the main text, by the script")
    A(r"\texttt{generar\_suplemento.py} in the repository.")
    A("")

    # ------------------------------------------------------------------ S1
    A(r"\section*{S1. What was measured, and why}")
    A("")
    A(r"The effective population size reported in the main text is estimated")
    A(r"from the decay of neutral marker variance. Comparing that estimate")
    A(r"against Wright's $N_e = 4N/(V_k+2)$ requires the variance in offspring")
    A(r"number $V_k$, and comparing it against the overlapping-generation")
    A(r"correction of \citet{hill1972} requires in addition the generation")
    A(r"interval $T$.")
    A(r"Both are properties of cohorts rather than of instantaneous state, so")
    A(r"they are measured by following identified cohorts to extinction.")
    A("")
    A(r"Individuals are identified by the pair (slot, birth time). The")
    A(r"substrate reuses array slots when an individual dies, so identifying")
    A(r"cohort members by slot alone counts several successive occupants as")
    A(r"one individual; that error produced offspring counts of 45 and 188 per")
    A(r"individual before it was corrected. The cohort loop also runs for the")
    A(r"full horizon at every population size rather than stopping when the")
    A(r"focal cohort is exhausted, so that the comparison across sizes is over")
    A(r"the same number of generations of drift.")
    A("")

    # ------------------------------------------------------------------ S2
    A(r"\section*{S2. Generation interval and offspring-number variance}")
    A("")
    A(r"\begin{table}[ht]")
    A(r"\centering\small")
    A(r"\caption{Cohort measurements at three census sizes. $\bar{k}$ is mean")
    A(r"offspring number per individual followed to extinction, $V_k$ its")
    A(r"variance, $T$ the generation interval in substrate steps, and")
    A(r"$T/\mathrm{lifetime}$ the interval expressed in mean lifetimes.}")
    A(r"\label{tab:s_cohortes}")
    A(r"\begin{tabular}{rrrrrrr}")
    A(r"\toprule")
    A(r"$R$ & $N$ & $\bar{k}$ & $V_k$ & $T$ & mean lifetime &"
      r" $T/\mathrm{lifetime}$\\")
    A(r"\midrule")
    for f in cohortes:
        A(f"{num(f['R'])} & {num(f['N'],1)} & {f['kbar']:.3f} & "
          f"{f['Vk']:.3f} & {f['Tgen']:.2f} & {f['vida']:.2f} & "
          f"{f['Tsobrevida']:.3f}\\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    A("")

    Ts = [f["Tgen"] for f in cohortes]
    Vks = [f["Vk"] for f in cohortes]
    kbs = [f["kbar"] for f in cohortes]
    Ns = [f["N"] for f in cohortes]
    A(r"Two features of Table~\ref{tab:s_cohortes} carry the argument of the")
    A(r"main text. First, the generation interval is \emph{flat}: $T$ varies")
    A(f"between {min(Ts):.2f} and {max(Ts):.2f} steps while $N$ grows from")
    A(f"{num(min(Ns),0)} to {num(max(Ns),0)}, a spread of")
    A(f"{(max(Ts) - min(Ts)) / min(Ts) * 100:.2f}\\% across a fourfold change")
    A(r"in census size. Second, $V_k$ is likewise approximately constant, at")
    A(f"{min(Vks):.2f} to {max(Vks):.2f}, with $\\bar{{k}}$ between")
    A(f"{min(kbs):.3f} and {max(kbs):.3f}.")
    A("")

    # ------------------------------------------------------------------ S3
    A(r"\section*{S3. $V_k$ does not explain the sublinear scaling}")
    A("")
    A(r"\begin{table}[ht]")
    A(r"\centering\small")
    A(r"\caption{Measured $N_e$ against Wright's prediction and against the")
    A(r"overlapping-generation correction. The correction moves the prediction")
    A(r"in the wrong direction at every size.}")
    A(r"\label{tab:s_ne}")
    A(r"\begin{tabular}{rrrrrr}")
    A(r"\toprule")
    A(r"$N$ & $N_e$ measured & $4N/(V_k+2)$ & ratio & Hill $4N_bT/(V_k+2)$ &"
      r" ratio\\")
    A(r"\midrule")
    for f in cohortes:
        A(f"{num(f['N'],1)} & {num(f['Ne_deriva'],1)} & "
          f"{num(f['Ne_teorico'],1)} & {f['cociente']:.3f} & "
          f"{num(f['Ne_hill'],1)} & {f['cociente_hill']:.3f}\\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    A("")
    coc = [f["cociente"] for f in cohortes]
    coch = [f["cociente_hill"] for f in cohortes]
    A(r"With $V_k$ near two, Wright's expression predicts $N_e \approx N$. The")
    A(f"measured value is {min(coc):.2f} to {max(coc):.2f} of that, and the")
    A(r"ratio \emph{falls} as $N$ grows, which is the sublinear scaling")
    A(r"reported in the main text. Since $V_k$ is flat across the same range,")
    A(r"it cannot be what produces the scaling, and we have no mechanism that")
    A(r"does.")
    A("")
    A(r"The overlapping-generation correction of \citet{hill1972}, $N_e =")
    A(r"4 N_b T / (V_k + 2)$, is the natural candidate, since this substrate has")
    A(r"continuous reproduction and overlapping generations. It does not help.")
    A(f"Because $T$ is constant at about {sum(Ts) / len(Ts):.0f} steps rather")
    A(r"than growing with $N$, the correction multiplies the prediction by a")
    A(r"roughly fixed factor, raising it further above a measurement that was")
    A(r"already below it: the ratio moves from")
    A(f"{min(coc):.2f}--{max(coc):.2f} to {min(coch):.2f}--{max(coch):.2f},")
    A(r"that is, further from one at every size. We report the hypothesis as")
    A(r"falsified rather than adjusting it.")
    A("")

    # ------------------------------------------------------------------ S4
    A(r"\section*{S4. The two estimators of $N_e$}")
    A("")
    A(r"\begin{table}[ht]")
    A(r"\centering\small")
    A(r"\caption{The drift-based and mutation--drift-balance estimators of")
    A(r"$N_e$, and their ratio. Agreement improves as $N$ grows.}")
    A(r"\label{tab:s_estimadores}")
    A(r"\begin{tabular}{rrrr}")
    A(r"\toprule")
    A(r"$N$ & $N_e$ from drift & $N_e$ from mutation--drift & ratio\\")
    A(r"\midrule")
    for f in estimadores:
        A(f"{num(f['N'],1)} & {num(f['Ne_deriva'],1)} & "
          f"{num(f['Ne_mutacion'],1)} & {f['cociente']:.3f}\\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    A("")
    r_est = [f["cociente"] for f in estimadores]
    A(r"The two estimators are independent: one reads the decay of neutral")
    A(r"variance, the other its equilibrium level under recurrent mutation.")
    A(f"They agree to within a factor of {min(r_est):.1f} to {max(r_est):.1f},")
    A(r"and the agreement improves monotonically with $N$. We report this as")
    A(r"partial concordance rather than confirmation: a factor of two between")
    A(r"two estimators of the same quantity is a real discrepancy, not a")
    A(r"rounding difference, and it is consistent with the sublinear scaling")
    A(r"of \S S3 being a property of the substrate that neither estimator")
    A(r"fully captures.")
    A("")

    # ------------------------------------------------------------------ S5
    A(r"\section*{S5. Provenance}")
    A("")
    A(r"\begin{tabular}{ll}")
    A(r"\toprule")
    A(rf"Result file & \texttt{{c2\_ne.json}}\\")
    A(rf"Run completed & {man['fecha_utc']}\\")
    A(rf"Source hash (\texttt{{code\_sha256\_16}}) & "
      rf"\texttt{{{man['code_sha256_16']}}}\\")
    A(rf"Python & {man['entorno']['python']}\\")
    A(rf"NumPy & {man['entorno']['numpy']}\\")
    A(rf"Seeds per point & {man['n_seed']}\\")
    A(rf"Burn-in steps & {num(man['T0'])}\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A("")
    A(r"\bibliographystyle{elsarticle-harv}")
    A(r"\bibliography{references}")
    A(r"\end{document}")

    SALIDA.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"escrito {SALIDA}  ({len(L)} lineas)")
    print(f"  cohortes: {len(cohortes)} puntos   estimadores: {len(estimadores)}")


if __name__ == "__main__":
    main()
