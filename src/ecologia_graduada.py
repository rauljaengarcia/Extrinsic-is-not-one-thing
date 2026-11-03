"""
Reglas de asignacion GRADUADAS, para la R1 (comentario R2.1 del Revisor 2).

Subclase de `ecologia.Ecologia`.  No toca `ecologia.py`: solo sustituye
`_transfer`, y cuando no se pide graduacion delega en el original.  La
regla manipulada de `ecologia.py` es binaria: el dominio de la regla parcial
excluye por completo a quien no lleva la etiqueta.  Aqui se gradua de dos
maneras, cada una con un parametro y siempre a partir de la regla PARCIAL
(`cond` = "Ip" o "Ep"):

  w    ADMISION FRACCIONADA, w en [0, 1].  Cada paso con transferencia, cada
       candidato que la regla parcial excluiria entra en el conjunto de
       candidatos con probabilidad w.  El sorteo usa un generador APARTE,
       derivado de la semilla ([seed, SAL_W]), de modo que el generador
       principal se consume exactamente igual que en `ecologia.py`.  El
       conjunto se construye con una mascara sobre `idx`, que conserva su
       orden.
         E: el asignador central atiende a L0 y admite a cada no-L0 con
            probabilidad w.
         I: cada donante atiende a su etiqueta y admite a cada candidato de la
            otra con probabilidad w.  Un mismo sorteo por candidato sirve a los
            dos grupos de donantes: con dos etiquetas, cada candidato es ajeno
            para uno solo de ellos.

  tau  SEGREGACION TEMPORAL, tau en [0, 1].  La regla es la parcial en una
       fraccion tau de los pasos y la abierta (la de A = 0) en el resto, con
       un calendario periodico y determinista: el paso t es parcial si
       t mod PERIODO < tau * PERIODO.  PERIODO = 20, mucho mas corto que la
       ventana de medida de 600 pasos: la ventana [10 000, 10 600) contiene 30
       periodos enteros, de modo que la fraccion parcial realizada en ella es
       exactamente tau; tau * 20 es entero para tau en {0.25, 0.5, 0.75}; y un
       donante aporta de media una vez cada 1/p_don = 50 pasos, asi que sus
       aportaciones sucesivas caen en uno u otro regimen sin seguir el
       calendario.

Extremos, comprobados bit a bit en `tests/test_r1.py`:

    E con w = 0  == E/p        E con w = 1 == E/x     (por la mascara, sin
    I con w = 0  == I/p                                despachar al original)
    tau = 0 == x  y  tau = 1 == p, en I y en E

I con w = 1 NO es I/x bit a bit, y no debe serlo: reparte con la misma regla
(todo donante puede elegir a cualquiera), pero en dos tandas, primero los
donantes L0 y luego los L1, mientras que I/x sortea a todos los donantes en
una sola tanda y en otro orden.  Se exige en cambio que kappa y s caigan dentro
del intervalo de I/x (`experiments/r1_graduada.py`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import runner                       # noqa: F401  (fija el threading antes de numpy)
import numpy as np

from ecologia import Ecologia
from runner import RunSpec

#: periodo del calendario de tau, en pasos
PERIODO = 20
#: sal del generador aparte de la admision fraccionada
SAL_W = 0x5257


class EcologiaGraduada(Ecologia):
    """Un mundo con la regla parcial graduada por `w` o por `tau` (no ambos)."""

    def __init__(self, cond="Ip", seed=0, cfg=None, remove_operator=False,
                 log_every=100, log_tree=False, w=None, tau=None, periodo=PERIODO):
        super().__init__(cond=cond, seed=seed, cfg=cfg,
                         remove_operator=remove_operator, log_every=log_every,
                         log_tree=log_tree)
        if w is not None and tau is not None:
            raise ValueError("w y tau son dos graduaciones distintas: una u otra")
        if (w is not None or tau is not None) and self.A != 1:
            raise ValueError("se gradua la regla parcial: cond tiene que ser Ip o Ep")
        for nombre, v in (("w", w), ("tau", tau)):
            if v is not None and not 0.0 <= v <= 1.0:
                raise ValueError(f"{nombre} fuera de [0, 1]: {v}")
        self.w = None if w is None else float(w)
        self.tau = None if tau is None else float(tau)
        self.periodo = int(periodo)
        self.k_parcial = None
        if self.tau is not None:
            k = self.tau * self.periodo
            if abs(k - round(k)) > 1e-9:
                raise ValueError(f"tau * periodo = {k} no es entero: la fraccion no seria exacta")
            self.k_parcial = int(round(k))
        self.rng_w = np.random.default_rng([self.seed, SAL_W]) if self.w is not None else None
        # diagnosticos, solo dentro de la ventana de medida
        self.ajenos_vistos = 0          # pares (grupo de donantes, candidato ajeno)
        self.ajenos_admitidos = 0
        self.pasos_parciales = 0        # pasos con transferencia, por regimen
        self.pasos_abiertos = 0

    # -- la regla graduada ------------------------------------------------
    def _transfer(self, idx):
        if self.w is None and self.tau is None:
            return super()._transfer(idx)
        c = self.cfg
        # Recaudacion y sorteo de donantes: identicos a `Ecologia._transfer`,
        # con el mismo consumo del generador principal.
        eligible = idx[self.E[idx] > c.E_don]
        if eligible.size == 0:
            return
        donors = eligible[self.rng.random(eligible.size) < c.p_don]
        if donors.size == 0:
            return
        self.collected += donors.size * c.q
        pools = self._pools_tau(donors, idx) if self.tau is not None else \
            self._pools_w(donors, idx)
        self._asignar(pools)

    def _pools_parciales(self, donors, idx):
        # las dos ramas parciales de `Ecologia._transfer`, tal cual
        if self.cond == "Ep":
            return [(donors, idx[self.L[idx] == 0])]
        pools = []
        for tag in (0, 1):
            d = donors[self.L[donors] == tag]
            if d.size:
                pools.append((d, idx[self.L[idx] == tag]))
        return pools

    def _pools_tau(self, donors, idx):
        parcial = (self.t % self.periodo) < self.k_parcial
        if self._acc_open:
            if parcial:
                self.pasos_parciales += 1
            else:
                self.pasos_abiertos += 1
        if not parcial:
            return [(donors, idx)]          # la rama abierta, la de A = 0
        return self._pools_parciales(donors, idx)

    def _pools_w(self, donors, idx):
        admitido = self.rng_w.random(idx.size) < self.w
        if self.cond == "Ep":
            propio = self.L[idx] == 0
            self._contar(propio, admitido)
            return [(donors, idx[propio | admitido])]
        pools = []
        for tag in (0, 1):
            d = donors[self.L[donors] == tag]
            if d.size:
                propio = self.L[idx] == tag
                self._contar(propio, admitido)
                pools.append((d, idx[propio | admitido]))
        return pools

    def _contar(self, propio, admitido):
        if self._acc_open:
            self.ajenos_vistos += int((~propio).sum())
            self.ajenos_admitidos += int((~propio & admitido).sum())

    def _asignar(self, pools):
        # el bucle de asignacion de `Ecologia._transfer`, tal cual
        c = self.cfg
        for d, pool in pools:
            if pool.size == 0:
                continue
            self.E[d] -= c.q
            pick = self.rng.integers(0, pool.size, (d.size, c.m_sample))
            cand = pool[pick]
            best = cand[np.arange(d.size), np.argmin(self.E[cand], axis=1)]
            np.add.at(self.E, best, c.q)
            self.transfers += d.size
            self.volume += d.size * c.q
            if self._acc_open:
                np.add.at(self.flow,
                          (self.L[d].astype(np.intp), self.L[best].astype(np.intp)),
                          c.q)

    def summary(self):
        out = super().summary()
        out.update({"w": self.w, "tau": self.tau, "periodo": self.periodo,
                    "ajenos_vistos": self.ajenos_vistos,
                    "ajenos_admitidos": self.ajenos_admitidos,
                    "pasos_parciales": self.pasos_parciales,
                    "pasos_abiertos": self.pasos_abiertos})
        return out


# --------------------------------------------------------------------------
# Especificacion y ejecucion, con la misma salida que `runner.run_one`
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpecGraduada(RunSpec):
    """`RunSpec` mas los dos parametros de la graduacion."""
    w: float | None = None
    tau: float | None = None
    periodo: int = PERIODO


def spec_graduada(**kw) -> RunSpecGraduada:
    if "cfg" in kw and isinstance(kw["cfg"], dict):
        kw["cfg"] = tuple(sorted(kw["cfg"].items()))
    return RunSpecGraduada(**kw)


def run_one_graduada(s: RunSpecGraduada) -> dict:
    """Como `runner.run_one`, con `EcologiaGraduada` en lugar de `Ecologia`."""
    e = EcologiaGraduada(cond=s.cond, seed=s.seed, cfg=s.config(),
                         remove_operator=s.remove_operator, log_every=0,
                         w=s.w, tau=s.tau, periodo=s.periodo)
    out = e.run(s.T, inject_at=s.inject_at, n_inject=s.n_inject,
                access_from=s.access_from, access_until=s.access_until)
    if s.access_from is not None:
        acc = e.access()
        for tag in (0, 1):
            out[f"recv_pc_L{tag}"] = float(acc["recv_pc"][tag])
            out[f"from_L0_pc_L{tag}"] = float(acc["from_L0_pc"][tag])
            out[f"prov_pc_L{tag}"] = float(acc["prov_pc"][tag])
            out[f"born_pc_L{tag}"] = float(acc["born_pc"][tag])
            out[f"expo_L{tag}"] = float(acc["expo"][tag])
        f = acc["flow"]
        out["flow_00"], out["flow_01"] = float(f[0, 0]), float(f[0, 1])
        out["flow_10"], out["flow_11"] = float(f[1, 0]), float(f[1, 1])
        cruzado = f[0, 1] + f[1, 0]
        out["flow_cross_frac"] = float(cruzado / f.sum()) if f.sum() > 0 else np.nan
    out["arm"] = s.arm
    out["label"] = s.label
    out["T"] = s.T
    return out


def _worker_graduado(payload):
    return run_one_graduada(RunSpecGraduada(**payload))


def run_grid_graduada(specs, workers=None, quiet=False):
    """`runner.run_grid` para `RunSpecGraduada`, con el mismo pool persistente."""
    import os
    import time
    specs = list(specs)
    workers = workers or min(12, os.cpu_count() or 1)
    t0 = time.perf_counter()
    payloads = [asdict(s) for s in specs]
    if workers == 1:
        rows = [_worker_graduado(p) for p in payloads]
    else:
        rows = runner._pool(workers).map(_worker_graduado, payloads, chunksize=1)
    dt = time.perf_counter() - t0
    if not quiet:
        print(f"  {len(specs)} runs en {dt:,.1f} s con {workers} workers "
              f"({len(specs)/dt:,.1f} runs/s)")
    return rows
