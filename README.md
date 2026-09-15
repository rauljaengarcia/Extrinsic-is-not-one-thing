# Extrinsic Is Not One Thing: Recipient Exchangeability in Evolutionary Systems

Code, data and reproduction material for the manuscript of the same name, by
Raúl-Antonio Jaén-García and Jorge Larrey-Ruiz (Universidad Politécnica de
Cartagena).

**This repository is administered by Raúl-Antonio Jaén-García.** It is private
until publication; access is granted on request to editors and reviewers —
write to `raul.jaen@edu.upct.es`.

The paper argues that a rule bearing on an evolutionary function has two
independent properties — who implements it, and whether what it supplies
depends on *who* the recipient is — and builds the smallest digital ecology in
which both can be manipulated independently. This repository contains
everything needed to re-derive every number, figure and table in it.

---

## 1. Reproduction

```bash
pip install -r requirements.txt

python reproduce.py tests     # regression suite — run this first
python reproduce.py all       # everything, in dependency order
```

`reproduce.py` is the single entry point. Individual steps:

| Step | Command | Produces | Approx. time |
|---|---|---|---|
| Regression suite | `python reproduce.py tests` | — | ~2 min |
| Substrate validation | `python reproduce.py validation` | `validacion.json` | ~8 min |
| The two interventions | `python reproduce.py orthogonality` | `ortogonalidad.json` | ~4 min |
| Three arms, 100 seeds | `python reproduce.py arms` | `brazos.json` | ~12 min |
| Emergent partiality (C1) | `python reproduce.py c1` | `c1_parcialidad_emergente.json` | ~3 min |
| V_k and N_e (C2) | `python reproduce.py c2` | `c2_ne.json` | ~6 min |
| Figures from existing JSON | `python reproduce.py figures` | `figures/*.png`, `*.pdf` | ~20 s |

Times are for a 12-core machine. Every step is deterministic: the same
specification produces the same output, bit for bit, regardless of how many
worker processes are used or in what order they finish. That is asserted by
the test suite, not just claimed here.

**One caveat about output directories.** Re-running an experiment writes to
`resultados/` and `figuras/`, not to `results/` and `figures/`. Those names are
hard-coded in `src/runner.py`, and `src/` is deliberately left byte-identical
to the tree that produced the published numbers — its digest is recorded in
every manifest, and renaming a directory there would invalidate all of them for
no scientific gain. Compare your fresh output against the published files;
they should match.

---

## 2. Which result file supports which figure and table

| Manuscript object | Result file | Produced by |
|---|---|---|
| Figure 1 — substrate validation | `results/validacion.json` | `experiments/validacion.py` |
| Figure 2 — orthogonality of the interventions | `results/ortogonalidad.json` | `experiments/exp_ortogonalidad.py` |
| Figure 3 — exclusion versus partition | `results/ortogonalidad.json` | `experiments/exp_ortogonalidad.py` |
| Figure 4 — the three arms | `results/brazos.json` | `experiments/exp_brazos.py` |
| Figure 5 — audited systems as profiles | *(none — transcribed from Table 2)* | `experiments/figuras.py` |
| Table 1 — the two coordinates are independent | *(none — definitional)* | — |
| Table 2 — the audit | *(none — read from published specifications)* | — |
| Table 3 — orthogonality of the interventions | `results/ortogonalidad.json` | `experiments/exp_ortogonalidad.py` |
| §6.1 substrate behaves like an evolving population | `results/validacion.json` | `experiments/validacion.py` |
| §6.2 no dynamical signature at baseline | `results/ortogonalidad.json` | `experiments/exp_ortogonalidad.py` |
| §6.4 exclusion versus partition | `results/ortogonalidad.json` | `experiments/exp_ortogonalidad.py` |
| §6.5 the range of the substitution contrast | `results/brazos.json` | `experiments/exp_brazos.py` |
| §6.6 does the open rule play favourites | `results/c1_parcialidad_emergente.json` | `experiments/analisis_c1.py` |
| Supplementary material — cohorts, N_e, Hill | `results/c2_ne.json` | `experiments/analisis_c2.py` |

Figure 5 has no result file on purpose: it is a literal transcription of
Table 2, with no aggregation and no numeric axes, so that the reader can check
it cell by cell against the table.

`results/brazos_30semillas.json` and its `.log` are the **archived 30-seed
run**, kept deliberately. It predates the confirmatory 100-seed batch and is
**not used anywhere in the manuscript**; it is here because a plateau that
appeared at 30 seeds turned out to be noise, and the record of that should
survive. Its manifest carries a different source digest, which is correct — it
was produced by an earlier state of the code.

---

## 3. Manifests and hashes

Every file in `results/` stores, alongside the data, a manifest recording:

- the complete run specifications — every condition, seed, horizon and
  configuration override;
- `code_sha256_16`, a digest of the source that produced it;
- the environment: Python version, NumPy version, platform, processor.

No result is separable from the conditions that generated it.

**`code_sha256_16` is a digest of all of `src/*.py`**, not of `ecologia.py`
alone. Comparing it against the hash of a single file will not match. To
recompute it:

```python
import sys; sys.path.insert(0, "src")
from runner import code_hash
print(code_hash())
```

---

## 4. What the test suite asserts

`python reproduce.py tests` checks each of the following, and fails loudly
rather than warning:

1. **Determinism.** A run specification determines its output bit for bit.
2. **Worker independence.** The output does not depend on the number of
   parallel worker processes, nor on the order in which they finish.
3. **Seeded bootstrap.** Resampling is reproducible.
4. **Monoculture identities.** The four experimental conditions are identical
   when every lineage carries the same label — the empirical content of the
   claim that neither coordinate has a dynamical signature at baseline.
5. **Orthogonality.** Each intervention moves exactly one coordinate.
6. **Substrate integrity.** Bounded genes never leave their range, the array
   capacity is never reached, and the declared-neutral marker is selectively
   neutral.
7. **Statistics.** Fisher's exact test, Wilson intervals, the `NA` guard on a
   null denominator, and bootstrap coverage.
8. **The dominance claims about Table 2.** `tests/dominancia.py` parses the
   audit table out of `manuscript/main.tex` — it does not transcribe it — and
   re-derives four properties the paper asserts, under *both* values the one
   indeterminate entry could take:
   - no artificial row dominates both the on-chain replicator's row and the
     sovereign agents' row;
   - the free-living microbial population dominates each of those two;
   - the genetic algorithm and the population-evolved language models are
     identical cell by cell, and are the only identical pair;
   - absent the intervention on maintenance, the sovereign agents would
     dominate the replicator on every function the evidence settles — and
     *not* on all six, which is why the manuscript qualifies the claim.

Item 8 exists because a property of a partial order over eight rows and six
functions is not something anyone reads correctly by eye, and twice a claim
reached the text that the table did not support.

---

## 5. Layout

```
├── LICENSE              Apache License 2.0 (code only — see §6)
├── NOTICE
├── README.md
├── CITATION.cff
├── requirements.txt
├── reproduce.py         single entry point
├── PREREGISTRO.md       pre-registration (in Spanish — see note below)
├── src/                 the substrate and its infrastructure
│   ├── ecologia.py      the digital ecology; the only file that evolves anything
│   ├── runner.py        run specifications, parallelism, manifests
│   ├── estadistica.py   bootstrap, Fisher, Wilson, the NA guard
│   └── estilo.py        figure style (CVD-safe palette, hatching)
├── experiments/         one script per published result
├── tests/               regression suite and the Table 2 checker
├── results/             the .json and .log files, each with its manifest
├── figures/             .png renderings
└── manuscript/          main.tex, references.bib, the five figures — see §7
```

**`PREREGISTRO.md` is in Spanish.** It is the pre-registration as it was
written before the experiments were run, and it is left in the original
language deliberately: translating a pre-registration after the fact is
exactly the kind of retrospective editing it exists to prevent. Section 9
records every deviation from the registered plan, including the ones that went
against the hypothesis.

Prototypes and calibration scripts from earlier stages (`eco.py`,
`invasion.py`, `calibrar_solo.py`, `calibrar2.py`) are **not** included: no
published result depends on them. `diag_theta.py` and `diag_calibrado.py` are
included, because the allocation-count threshold declared in the
pre-registration is justified by what `diag_theta.py` measures.

---

## 6. Licensing — the code and the manuscript are covered differently

The source code in this repository is licensed under the Apache License 2.0
(see `LICENSE`). The manuscript, its figures and its text are **not** covered
by that licence and remain all rights reserved by the author until
publication, after which they are governed by the terms agreed with the
publisher.

---

## 7. The manuscript is here so the suite can check itself

`manuscript/` contains the LaTeX source of the paper. It is present for one
reason: `tests/dominancia.py` parses Table 2 out of `main.tex`. Without it
that part of the suite skips, and the repository stops verifying what §4 above
says it verifies.

It is not covered by Apache 2.0, and it can be removed before the repository
is made public if the agreement with the publisher makes that advisable —
the suite degrades gracefully, reporting the check as skipped rather than
failing.

---

## 8. How to cite

See `CITATION.cff`. Cite the manuscript for the ideas and results; cite the
software entry for the code itself.
