"""
Ecologia de organismos digitales con genoma vectorial.

Sustrato para el experimento de dos coordenadas (control x recipient
exchangeability).  Sustituye al prototipo `experimento/eco.py`, que tenia dos
genes decorativos, ningun compromiso real, y un observable (supervivencia a T)
que resulto ser un artefacto de promediado.

--------------------------------------------------------------------------
EL GENOMA: cuatro genes, cada uno con un papel declarado y una prediccion
--------------------------------------------------------------------------

    g[0]  a      afinidad de captacion del recurso.
                 El reparto del recurso es proporcional a `a` (suma cero) y el
                 coste de mantenimiento es c0 + kc*a^2.  El termino cuadratico
                 acota la carrera armamentistica y produce un OPTIMO INTERIOR
                 analitico (ESS), que es la prediccion fuerte de la validacion:

                     a* = sqrt( R*(N-1) / (2*kc*N^2) )

    g[1]  phi    fraccion de la energia del progenitor que recibe la
                 descendencia.  Compromiso de historia vital: phi bajo deja
                 descendencia pobre (mortalidad dependiente de condicion alta),
                 phi alto deja al progenitor pobre y retrasa su siguiente
                 evento reproductivo.  El compromiso solo tiene optimo interior
                 si acumular reservas cuesta algo: de ahi el termino `ks*E` del
                 coste de mantenimiento.  Sin el, phi se dispara al tope porque
                 el progenitor puede esperar indefinidamente sin riesgo.

    g[2]  m      modificador de la tasa de mutacion propia:
                 sigma_i = mu_base * exp(m_i).  Afecta a la varianza de la
                 descendencia.  Prediccion: principio de reduccion, deriva a la
                 baja cerca del optimo.

    g[3]  z      MARCADOR SELECTIVAMENTE NEUTRAL.  No es un gen decorativo: es
                 el instrumento de medida.  Con el se miden la deriva
                 (decaimiento de la varianza como (1-1/2N)^t) y la respuesta a
                 la seleccion frente a la ecuacion del criador R = h^2 S.
                 Su neutralidad es una propiedad declarada y comprobada, no un
                 descuido.

--------------------------------------------------------------------------
LA REGLA MANIPULADA: transferencia de energia
--------------------------------------------------------------------------

Las cuatro condiciones son cuatro versiones de UNA MISMA regla de asignacion,
que difieren en exactamente una cosa cada una respecto de sus vecinas.

Recaudacion (identica en las cuatro): todo organismo con E > E_don aporta `q`.
Asignacion:   cada aportacion va al candidato mas necesitado (menor E) de una
              muestra aleatoria de tamano `m_sample`.

              operador = los propios organismos   -> C = I
              operador = un asignador central     -> C = E

              muestra = cualquier organismo vivo  -> A = 0  (exchangeable)
              muestra = solo la etiqueta L0       -> A = 1  (recipient-partial)
                        (en I/p: solo la etiqueta del propio donante)

    I/x   intrinseco, exchangeable
    I/p   intrinseco, recipient-partial   (cuidado dirigido a la propia estirpe)
    E/x   extrinseco, exchangeable
    E/p   extrinseco, recipient-partial

Volumen, momento, criterio, tamano de muestra y topologia son IDENTICOS en las
cuatro celdas por construccion.  Lo unico que cambia es quien ejecuta la
transferencia y si el dominio de la regla incluye la etiqueta.

--------------------------------------------------------------------------
LAS DOS INTERVENCIONES
--------------------------------------------------------------------------

    retirada del operador   `remove_operator=True`
        Retira el soporte suministrado DESDE FUERA de D.  En C=E desaparece la
        transferencia.  En C=I no hay nada externo que retirar y la
        transferencia continua: el operador es parte del sistema focal.

    reetiquetado            `inject(t_inj, n, tag=1)`
        Clona organismos vivos copiando su estado EXACTAMENTE y cambiandoles
        solo la etiqueta.  En A=0 los clones reciben lo mismo que los demas.
        En A=1 no reciben nada.

Cada intervencion responde a exactamente una coordenada.  Esa ortogonalidad
es el resultado, y es tambien la razon por la que en linea base las cuatro
condiciones pueden ser indistinguibles: ninguna coordenada tiene firma
dinamica sin la intervencion que la revela.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------

#: Nombres de los genes, en orden.
GENES = ("a", "phi", "m", "z")

#: Limites de cada gen.  El marcador neutral no se acota: debe poder derivar.
GENE_BOUNDS = {
    "a":   (0.05, 10.0),
    "phi": (0.02, 0.98),
    "m":   (-3.0, 3.0),
    "z":   (-np.inf, np.inf),
}

CONDICIONES = ("Ix", "Ip", "Ex", "Ep")


class Config:
    """Parametros del mundo.  Todo valor arbitrario vive aqui y se registra en
    el manifiesto de cada run."""

    __slots__ = ("R", "c0", "kc", "ks", "theta", "c_rep", "d_min", "d_max",
                 "E_s", "E_don", "q", "p_don", "m_sample", "mu_base", "n0",
                 "n0_L1", "cap", "E_init", "g_init", "g_init_sd", "transfer_on",
                 "mutate_genes",
                 "provider", "Q_rate", "Q_total", "match", "c_enrol",
                 "suppress", "suppress_frac", "focal_tag",
                 "select_on", "select_frac")

    def __init__(self, **kw):
        # --- recurso y coste
        self.R        = 500.0   # recurso total repartido por paso
        self.c0       = 0.18    # coste de mantenimiento basal
        self.kc       = 0.25    # coeficiente del coste cuadratico de la afinidad
        self.ks       = 0.02    # coste de almacenamiento por unidad de reserva
        # --- reproduccion
        self.theta    = 10.0    # umbral reproductivo
        self.c_rep    = 1.0     # sobrecoste fijo de un evento reproductivo
        # --- mortalidad dependiente de condicion
        self.d_min    = 0.001   # riesgo asintotico con energia alta
        self.d_max    = 0.050   # riesgo con energia nula
        self.E_s      = 5.0     # escala de energia del riesgo
        # --- la regla manipulada
        self.transfer_on = True
        self.E_don    = 7.5     # umbral de aportacion
        self.q        = 3.00    # aportacion por evento (suma unica, no goteo)
        self.p_don    = 0.02    # probabilidad de aportar por donante y paso
        self.m_sample = 8       # candidatos muestreados por aportacion
        # --- el proveedor de recurso reproductivo (maquinaria de los 3 brazos)
        #     'none'    brazo R: sin proveedor
        #     'partial' brazo B: solo atiende a la etiqueta focal    -> A = 1
        #     'open'    brazo S: atiende a quien cumpla una condicion
        #               DINAMICAMENTE ALCANZABLE (inscribirse pagando) -> A = 0
        self.provider = "none"
        self.Q_rate   = 0.00    # bonus por receptor y paso   (igualado per capita)
        self.Q_total  = 0.00    # bonus total por paso        (igualado por volumen)
        self.match    = "per_capita"   # 'per_capita' | 'total'
        self.c_enrol  = 0.5     # coste de inscripcion en el brazo abierto.
                                # Se informa POR SEPARADO, nunca absorbido en Phi.
        # --- la supresion de competidores (la manipulacion que da rho ~ 1)
        #     'partial' veta la reproduccion de quien no lleva la etiqueta focal
        #     'open'    veta la fraccion inferior por energia, simetricamente
        self.suppress = "none"
        self.suppress_frac = 0.0    # calibrado en pilotos descartados
        self.focal_tag = 0
        # --- seleccion por truncamiento impuesta desde fuera
        #     Es a la vez el gancho de la validacion (ecuacion del criador) y
        #     una condicion real del marco: una funcion de fitness impuesta es
        #     extrinseca y perfectamente exchangeable.  Permutar etiquetas no
        #     cambia ninguna asignacion; solo el rasgo decide.
        self.select_on = None       # None o indice de gen en GENES
        self.select_frac = 1.0      # fraccion superior que puede reproducirse
        # --- mutacion
        self.mu_base  = 0.01    # desviacion tipica base, modulada por g[2]
        self.mutate_genes = (0, 1, 2, 3)   # que genes mutan.
        # Restringirlo es lo que permite medir DERIVA y no arrastre: con los
        # genes fisiologicos congelados e identicos entre individuos no hay
        # ninguna seleccion de la que el marcador pueda hitchhike, y lo que
        # queda es muestreo puro.
        # --- poblacion
        self.n0       = 400
        self.n0_L1    = 0       # cuantos de los iniciales llevan la etiqueta L1.
                                # La parcialidad necesita a quien desfavorecer:
                                # en monocultivo no tiene consecuencia dinamica.
        self.cap      = 6000    # tamano de los arrays; no debe llegar a morder
        self.E_init   = 5.0
        self.g_init   = (1.0, 0.5, 0.0, 0.0)
        self.g_init_sd = 0.05
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(f"parametro desconocido: {k}")
            setattr(self, k, v)

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


# --------------------------------------------------------------------------
# El mundo
# --------------------------------------------------------------------------

class Ecologia:
    """Un mundo.  Una semilla, una condicion, una configuracion.

    Reproducibilidad exacta: misma semilla y misma configuracion producen la
    misma trayectoria bit a bit.  El orden de consumo del generador es
    identico en las cuatro condiciones, que es lo que permite los tests de
    regresion de monocultivo.
    """

    def __init__(self, cond="Ix", seed=0, cfg=None, remove_operator=False,
                 log_every=100, log_tree=False):
        if cond not in CONDICIONES:
            raise ValueError(f"condicion desconocida: {cond}")
        self.cond = cond
        self.seed = int(seed)
        self.cfg = cfg or Config()
        self.remove_operator = bool(remove_operator)
        self.log_every = int(log_every)
        self.rng = np.random.default_rng(seed)

        c = self.cfg
        cap = c.cap
        self.G = np.zeros((cap, 4), dtype=np.float64)
        self.E = np.zeros(cap, dtype=np.float64)
        self.L = np.zeros(cap, dtype=np.int8)
        self.alive = np.zeros(cap, dtype=bool)
        self.birth_t = np.full(cap, -1, dtype=np.int32)
        self.enrolled = np.zeros(cap, dtype=bool)

        n0 = c.n0
        # `g_init_sd` admite un escalar o un vector por gen.  El vector hace
        # falta para aislar la deriva: si el marcador arranca con varianza a la
        # vez que un gen bajo seleccion fuerte, lo que se mide no es deriva sino
        # ARRASTRE GENETICO, y decae mucho mas deprisa y de forma erratica.
        sd = np.broadcast_to(np.asarray(c.g_init_sd, dtype=np.float64), (4,))
        self.G[:n0] = np.asarray(c.g_init) + self.rng.normal(0, 1, (n0, 4)) * sd
        for i, name in enumerate(GENES):
            lo, hi = GENE_BOUNDS[name]
            if np.isfinite(lo) or np.isfinite(hi):
                np.clip(self.G[:n0, i], lo, hi, out=self.G[:n0, i])
        self.E[:n0] = c.E_init
        self.alive[:n0] = True
        if c.n0_L1:
            # los L1 son estadisticamente indistinguibles de los L0: mismo
            # genoma inicial, misma energia. Solo cambia la etiqueta.
            self.L[self.rng.choice(n0, int(c.n0_L1), replace=False)] = 1

        self.t = 0
        self.births = 0
        self.deaths = 0
        self.transfers = 0          # numero de aportaciones efectivamente asignadas
        self.volume = 0.0           # energia total transferida
        self.collected = 0.0        # energia total recaudada (para el igualado)
        self.cap_bound_hits = 0     # veces que la capacidad del array ha mordido
        self.provided = 0.0         # volumen total emitido por el proveedor
        self.acq_cost = 0.0         # coste de adquisicion pagado por los receptores;
                                    # se informa aparte, nunca absorbido en Phi
        self.n_enrol = 0            # inscripciones
        self.suppressed = 0         # eventos reproductivos vetados
        self.births_tag = np.zeros(2, dtype=np.int64)   # nacimientos por etiqueta

        # --- observable del test de reetiquetado ---------------------------
        # `flow[i, j]` = energia que va de donantes con etiqueta i a receptores
        # con etiqueta j.  La matriz completa hace falta porque las dos celdas
        # parciales NO son iguales en consecuencia agregada:
        #
        #   E/p  un asignador central condicionado a la identidad  -> EXCLUSION
        #        L1 no recibe de nadie.
        #   I/p  cada donante atiende a los de su propia etiqueta  -> PARTICION
        #        L1 recibe, pero solo de si mismo.
        #
        # Ambas son A=1 (reetiquetar cambia la asignacion) y difieren por
        # completo en efecto. El invariante que las une es el flujo CRUZADO:
        # bajo A=1 es exactamente cero; bajo A=0 sigue a la composicion.
        self.flow = np.zeros((2, 2), dtype=np.float64)
        self.prov_recv = np.zeros(2, dtype=np.float64)  # recibido del proveedor
        self.expo = np.zeros(2, dtype=np.float64)       # organismo-pasos por etiqueta
        self.born = np.zeros(2, dtype=np.int64)         # nacimientos por etiqueta
        self._acc_open = False                          # ventana de medida abierta

        self.series = []
        self._tree = [] if log_tree else None
        # ultimos progenitores y descendientes, para medir S y R paso a paso
        self.last_pool = np.zeros(0, dtype=np.intp)
        self.last_candidates = np.zeros(0, dtype=np.intp)
        self.last_parents = np.zeros(0, dtype=np.intp)
        self.last_offspring = np.zeros(0, dtype=np.intp)

    # -- propiedades del operador ------------------------------------------
    @property
    def C(self):
        """Coordenada de control de la regla manipulada."""
        return "I" if self.cond[0] == "I" else "E"

    @property
    def A(self):
        """Coordenada de recipient exchangeability de la regla manipulada."""
        return 0 if self.cond[1] == "x" else 1

    @property
    def transfer_active(self):
        """La transferencia ocurre salvo que se retire un operador externo.

        En C=I no hay operador externo que retirar: el mecanismo esta
        instanciado dentro de D y retirarlo seria amputar el sistema focal, no
        retirarle soporte.  Esta asimetria ES la coordenada de control.
        """
        if not self.cfg.transfer_on:
            return False
        if self.remove_operator and self.C == "E":
            return False
        return True

    # -- dinamica -----------------------------------------------------------
    def _uptake(self, idx):
        c = self.cfg
        a = self.G[idx, 0]
        share = a / a.sum()
        income = c.R * share
        # El termino ks*E penaliza acumular reservas.  No depende de `a`, luego
        # no altera el gradiente de invasion y la prediccion ESS sigue en pie.
        cost = c.c0 + c.kc * a * a + c.ks * np.maximum(self.E[idx], 0.0)
        return income - cost

    def _transfer(self, idx):
        """La regla manipulada.  Identica en las cuatro condiciones salvo en
        quien la ejecuta y en si el dominio incluye la etiqueta."""
        c = self.cfg
        eligible = idx[self.E[idx] > c.E_don]
        if eligible.size == 0:
            return
        # El sorteo de quien aporta se hace ANTES de partir por etiqueta, con
        # un numero de extracciones que no depende de la condicion.  Es lo que
        # permite que las cuatro celdas consuman el generador en el mismo orden
        # y que los tests de regresion de monocultivo sean bit a bit.
        donors = eligible[self.rng.random(eligible.size) < c.p_don]
        if donors.size == 0:
            return
        self.collected += donors.size * c.q

        if self.A == 0:
            # exchangeable: el conjunto de candidatos es toda la poblacion
            pools = [(donors, idx)]
        elif self.cond == "Ep":
            # extrinseco y parcial: el asignador central solo atiende a L0
            pool = idx[self.L[idx] == 0]
            pools = [(donors, pool)]
        else:  # Ip
            # intrinseco y parcial: cada donante solo atiende a su propia
            # etiqueta.  Es el analogo del cuidado dirigido a la estirpe.
            pools = []
            for tag in (0, 1):
                d = donors[self.L[donors] == tag]
                if d.size:
                    pools.append((d, idx[self.L[idx] == tag]))

        for d, pool in pools:
            if pool.size == 0:
                continue
            self.E[d] -= c.q
            # cada aportacion elige al mas necesitado de una muestra aleatoria
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

    def _provider(self, idx):
        """Proveedor de recurso reproductivo.  Es la regla sobre la que se
        corren los tres brazos B / R / S.

        El brazo abierto exige una condicion DINAMICAMENTE ALCANZABLE
        (inscribirse pagando `c_enrol`), no una propiedad que solo el linaje
        focal pueda satisfacer.  Si la condicion no fuera alcanzable, el brazo
        no habria des-personalizado nada y `rho` mediria otra cosa.
        """
        c = self.cfg
        if c.provider == "none":
            return
        if c.provider == "partial":
            recv = idx[self.L[idx] == c.focal_tag]
        else:  # open
            # inscripcion: cualquiera con energia suficiente puede pagarla
            cand = idx[(~self.enrolled[idx]) & (self.E[idx] > c.c_enrol + c.theta * 0.2)]
            if cand.size:
                self.E[cand] -= c.c_enrol
                self.enrolled[cand] = True
                self.acq_cost += cand.size * c.c_enrol
                self.n_enrol += cand.size
            recv = idx[self.enrolled[idx]]
        if recv.size == 0:
            return
        if c.match == "per_capita":
            amount = c.Q_rate
            total = amount * recv.size
        else:                       # 'total': volumen emitido constante
            total = c.Q_total
            amount = total / recv.size
        self.E[recv] += amount
        self.provided += total
        if self._acc_open:
            np.add.at(self.prov_recv, self.L[recv].astype(np.intp), amount)

    def _suppression_mask(self, idx, ok):
        """Devuelve la mascara de quien NO puede reproducirse.

        'partial' veta por etiqueta: el beneficio de la regla es SER el exento.
        Eso es lo que por construccion no admite sustituto exchangeable: una
        regla abierta que vetase "a quien cumpla X" vetaria tambien al linaje
        focal, luego no puede reproducir el beneficio de la exencion.

        'open' veta la fraccion inferior por energia DENTRO DEL CONJUNTO
        ELEGIBLE.  Calcular el cuantil sobre la poblacion entera haria la regla
        vacua: quien se reproduce tiene E > theta, muy por encima de la mediana,
        y el veto no alcanzaria a nadie.
        """
        c = self.cfg
        if c.suppress == "none":
            return np.zeros(idx.size, dtype=bool)
        if c.suppress == "partial":
            return self.L[idx] != c.focal_tag
        if not ok.any() or c.suppress_frac <= 0.0:
            return np.zeros(idx.size, dtype=bool)
        cut = np.quantile(self.E[idx][ok], c.suppress_frac)
        return self.E[idx] < cut

    def _mortality(self, idx):
        c = self.cfg
        p = c.d_min + (c.d_max - c.d_min) * np.exp(-np.maximum(self.E[idx], 0.0) / c.E_s)
        u = self.rng.random(idx.size)
        return idx[u < p]

    def _reproduce(self, idx):
        c = self.cfg
        phi = self.G[idx, 1]
        ok = (self.E[idx] > c.theta) & ((1.0 - phi) * self.E[idx] - c.c_rep > 0.0)
        if c.suppress != "none":
            blocked = self._suppression_mask(idx, ok)
            self.suppressed += int((ok & blocked).sum())
            ok = ok & ~blocked
        # El conjunto de CANDIDATOS es la referencia correcta para S y R: es el
        # conjunto sobre el que actua la seleccion.  Medir el diferencial contra
        # la poblacion entera lo diluye por un factor que depende de la
        # demografia, y entonces R/S deja de ser 1 sin que nada este mal.
        self.last_candidates = idx[ok]
        if c.select_on is not None and c.select_frac < 1.0 and ok.any():
            v = self.G[idx, c.select_on]
            cut = np.quantile(v[ok], 1.0 - c.select_frac)
            ok = ok & (v >= cut)
        parents = idx[ok]
        self.last_pool = idx
        self.last_parents = parents
        self.last_offspring = np.zeros(0, dtype=np.intp)
        if parents.size == 0:
            return
        free = np.flatnonzero(~self.alive)
        k = min(free.size, parents.size)
        if k < parents.size:
            # La capacidad del array no debe morder nunca: si muerde, elegir a
            # los progenitores por orden de indice introduciria un sesgo
            # correlacionado con la antiguedad, que es justo el artefacto que
            # este trabajo estudia.  Se elige al azar y se contabiliza.
            self.cap_bound_hits += 1
            parents = self.rng.choice(parents, k, replace=False)
        if k == 0:
            return
        p, f = parents[:k], free[:k]
        self.last_parents, self.last_offspring = p, f
        phi_p = self.G[p, 1]
        endow = phi_p * self.E[p]
        sigma = self.cfg.mu_base * np.exp(self.G[p, 2])
        paso = self.rng.normal(0.0, 1.0, (k, 4)) * sigma[:, None]
        if len(c.mutate_genes) < 4:
            m = np.zeros(4)
            m[list(c.mutate_genes)] = 1.0
            paso *= m
        self.G[f] = self.G[p] + paso
        # Los genes acotados se recortan AL NACER, de modo que el valor
        # almacenado es el valor expresado.  Si no, el gen deriva neutralmente
        # por encima de su tope y la media informada deja de significar nada.
        for i, name in enumerate(GENES):
            lo, hi = GENE_BOUNDS[name]
            if np.isfinite(lo) or np.isfinite(hi):
                np.clip(self.G[f, i], lo, hi, out=self.G[f, i])
        self.E[f] = endow
        self.E[p] = (1.0 - phi_p) * self.E[p] - c.c_rep
        self.L[f] = self.L[p]
        self.birth_t[f] = self.t
        self.alive[f] = True
        self.enrolled[f] = False        # la inscripcion no se hereda
        self.births += k
        np.add.at(self.births_tag, self.L[f].astype(np.intp), 1)
        if self._acc_open:
            np.add.at(self.born, self.L[f].astype(np.intp), 1)
        if self._tree is not None:
            self._tree.append(np.stack([
                np.full(k, self.t, dtype=np.int32), p.astype(np.int32),
                f.astype(np.int32), self.L[f].astype(np.int32)], axis=1))

    def step(self):
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return False

        if self._acc_open:
            np.add.at(self.expo, self.L[idx].astype(np.intp), 1.0)

        self.E[idx] += self._uptake(idx)

        self._provider(idx)

        if self.transfer_active:
            self._transfer(idx)

        dead = self._mortality(idx)
        starved = idx[self.E[idx] <= 0.0]
        if dead.size or starved.size:
            gone = np.union1d(dead, starved)
            self.alive[gone] = False
            self.deaths += gone.size
            idx = np.flatnonzero(self.alive)
            if idx.size == 0:
                self.t += 1
                return False

        self._reproduce(idx)

        self.t += 1
        if self.log_every and self.t % self.log_every == 0:
            self._snapshot()
        return bool(self.alive.any())

    def _snapshot(self):
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            self.series.append((self.t, 0, *([np.nan] * 4), np.nan,
                                self.births, self.deaths, 0, 0))
            return
        g = self.G[idx].mean(axis=0)
        self.series.append((
            self.t, idx.size, g[0], g[1], g[2], g[3], self.E[idx].mean(),
            self.births, self.deaths,
            int((self.L[idx] == 0).sum()), int((self.L[idx] == 1).sum())))

    # -- intervenciones -----------------------------------------------------
    def inject(self, n, tag=1):
        """Reetiquetado: clona `n` organismos vivos copiando su estado
        exactamente y cambiandoles solo la etiqueta."""
        live = np.flatnonzero(self.alive)
        if live.size == 0:
            return 0
        src = self.rng.choice(live, min(n, live.size), replace=False)
        free = np.flatnonzero(~self.alive)[:src.size]
        k = min(free.size, src.size)
        if k == 0:
            return 0
        self.G[free[:k]] = self.G[src[:k]]      # estado identico
        self.E[free[:k]] = self.E[src[:k]]
        self.birth_t[free[:k]] = self.t
        self.enrolled[free[:k]] = self.enrolled[src[:k]]   # la inscripcion es estado
        self.L[free[:k]] = tag                  # solo cambia la etiqueta
        self.alive[free[:k]] = True
        return k

    def open_access_window(self):
        """Abre la ventana de medida del observable de acceso."""
        self.flow[:] = 0.0
        self.prov_recv[:] = 0.0
        self.expo[:] = 0.0
        self.born[:] = 0
        self._acc_open = True

    def access(self):
        """Acceso per capita por etiqueta dentro de la ventana.

        Es el observable correcto del test de reetiquetado: mide lo que la
        REGLA ASIGNA, no quien sobrevive a la deriva.  La supervivencia a
        horizonte largo es un estimador con absorcion, y su media entre
        semillas es indistinguible de la deriva neutral: fue el artefacto que
        invalido la medida del prototipo anterior.

        `from_L0` es el estadistico decisivo.  Es el dominio literal de la
        regla: lo que los donantes de la poblacion residente asignan a cada
        etiqueta.  Bajo A=0 es igual entre etiquetas; bajo A=1 es cero para la
        etiqueta ajena, tanto si la regla excluye (E/p) como si particiona
        (I/p).
        """
        expo = np.maximum(self.expo, 1.0)
        total = self.flow.sum(axis=0) + self.prov_recv
        return {
            "recv_pc": total / expo,                    # todo lo recibido
            "from_L0_pc": self.flow[0] / expo,          # recibido de donantes L0
            "prov_pc": self.prov_recv / expo,           # recibido del proveedor
            "born_pc": self.born / expo,
            "flow": self.flow.copy(),
            "expo": self.expo.copy(),
        }

    # -- ejecucion ----------------------------------------------------------
    def run(self, T, inject_at=None, n_inject=0, access_from=None,
            access_until=None):
        """`access_until` cierra la ventana de medida del acceso.

        La ventana debe ser CORTA.  Los clones se inyectan con el estado de sus
        originales, pero en cuanto la simulacion avanza sus estados divergen, y
        una regla exchangeable que asigna por necesidad reparte entonces de
        forma distinta a los dos grupos por razones de ESTADO, no de etiqueta.
        Medir en una ventana larga mezcla el efecto de la regla con la deriva
        de los estados y ensucia el cociente que deberia valer 1.
        """
        for _ in range(T):
            if inject_at is not None and self.t == inject_at:
                self.inject(n_inject)
            if access_from is not None and self.t == access_from:
                self.open_access_window()
            if access_until is not None and self.t == access_until:
                self._acc_open = False
            if not self.step():
                break
        return self.summary()

    def summary(self):
        idx = np.flatnonzero(self.alive)
        out = {
            "cond": self.cond, "seed": self.seed, "t": self.t,
            "remove_operator": self.remove_operator,
            "births": self.births, "deaths": self.deaths,
            "n_final": int(idx.size),
            "transfers": self.transfers, "volume": self.volume,
            "collected": self.collected,
            "cap_bound_hits": self.cap_bound_hits,
            "extinct": bool(idx.size == 0),
            # Phi primaria: ciclos reproductivos autonomos del linaje focal
            "phi_focal": int(self.births_tag[self.cfg.focal_tag]),
            "births_L0": int(self.births_tag[0]),
            "births_L1": int(self.births_tag[1]),
            # el coste de adquisicion se informa APARTE, nunca dentro de Phi
            "provided": self.provided, "acq_cost": self.acq_cost,
            "n_enrol": self.n_enrol, "suppressed": self.suppressed,
        }
        if idx.size:
            g = self.G[idx]
            out.update({f"g_{n}": float(g[:, i].mean()) for i, n in enumerate(GENES)})
            out.update({f"sd_{n}": float(g[:, i].std()) for i, n in enumerate(GENES)})
            out["E_mean"] = float(self.E[idx].mean())
            out["n_L0"] = int((self.L[idx] == 0).sum())
            out["n_L1"] = int((self.L[idx] == 1).sum())
        else:
            out.update({f"g_{n}": np.nan for n in GENES})
            out.update({f"sd_{n}": np.nan for n in GENES})
            out["E_mean"] = np.nan
            out["n_L0"] = 0
            out["n_L1"] = 0
        return out

    def tree(self):
        """Arbol de linaje: (t, progenitor, descendiente, etiqueta)."""
        if self._tree is None:
            raise RuntimeError("este run no registro el arbol (log_tree=False)")
        if not self._tree:
            return np.zeros((0, 4), dtype=np.int32)
        return np.concatenate(self._tree, axis=0)

    def series_array(self):
        return np.asarray(self.series, dtype=np.float64)


# --------------------------------------------------------------------------
# Predicciones analiticas, para la validacion del sustrato
# --------------------------------------------------------------------------

def ess_afinidad(cfg, N):
    """Optimo evolutivamente estable de la afinidad.

    El ingreso de i es R*a_i/sum(a_j) y su coste kc*a_i^2.  Igualando el
    gradiente de invasion a cero en una poblacion monomorfica de tamano N:

        d/da_i [ R*a_i/(sum a) ]  =  R*(N-1)*a / (N*a)^2  =  R*(N-1)/(N^2*a)
        d/da_i [ kc*a_i^2 ]       =  2*kc*a

        =>  a* = sqrt( R*(N-1) / (2*kc*N^2) )
    """
    N = np.asarray(N, dtype=np.float64)
    return np.sqrt(cfg.R * (N - 1.0) / (2.0 * cfg.kc * N * N))
