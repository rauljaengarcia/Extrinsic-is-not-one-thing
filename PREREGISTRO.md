# Pre-registro — experimento de dos coordenadas

Fijado **antes** de ejecutar los runs confirmatorios. El calibrado del igualado
se hace sobre **semillas piloto disjuntas** de las confirmatorias y se descarta.

Código en el estado con hash `code_sha256_16` que aparece en el manifiesto de
cada fichero de `resultados/`.

---

## 1. Sistema focal y división estado/etiqueta

**D** = la población de organismos que lleva la etiqueta focal `L0`.

- **estado `z`**: genoma (afinidad `a`, dotación `phi`, modificador de mutación
  `m`, marcador neutral `z`), energía acumulada, inscripción en el proveedor
  abierto.
- **etiqueta `ℓ`**: el *lineage tag*. Nada que un organismo pueda hacer lo
  cambia; solo registra de qué siembra procede.

La inscripción es **estado** y no etiqueta porque es *dinámicamente
alcanzable*: cualquier organismo con energía suficiente puede pagarla. Es lo
que hace que el brazo S des-personalice realmente la regla.

## 2. Horizonte y medida de continuidad

- `T = 20 000` pasos en todos los brazos.
- **Φ primaria**: ciclos reproductivos autónomos completados por el linaje
  focal dentro de `T` (`phi_focal`).
- Φ secundarias, calculadas post hoc sobre el mismo registro, para informar
  robustez a la elección de Φ:
  1. probabilidad de persistencia del linaje focal a `T`;
  2. variación heredable retenida (varianza genómica final).
- **No** se informa rendimiento de optimización como resultado.

## 3. Réplicas

- 30 semillas por celda en los experimentos confirmatorios (`seed = 0..29`).
- Pilotos de calibrado: `seed = 1000..1015`. **Se descartan.**
- Los dos rangos son disjuntos por construcción.

## 4. Estadística

- Resultado primario: `Φ_B`, `Φ_R`, `Φ_S` **crudos** con intervalos bootstrap
  percentil, `n_boot = 10 000`, α = 0,05, semilla del bootstrap `20260911`.
- `δ` y `ρ` son **derivados**, con intervalos propagados por bootstrap sobre
  las muestras de cada brazo por separado.
- **Guardián**: `ε = 0,05 · Φ_B` de cada manipulación. `ρ := NA` cuando
  `Φ_B − Φ_R < ε` o cuando el intervalo de `Φ_B − Φ_R` cubre 0.
- Donde el guardián muerde la salida es **`NA`, nunca cero**. Asignar cero
  registraría una regla parcial como exchangeable.

## 5. Tolerancia de igualado

El brazo S debe igualar al brazo B en toda propiedad causal salvo la identidad.

- **Volumen emitido** por el proveedor: tolerancia **±5 %** sobre el total
  emitido por paso, calibrada en los pilotos.
- **Fracción vetada** en la manipulación de supresión: tolerancia **±5 %**
  sobre la proporción de individuos elegibles vetados.
- **Momento, latencia y topología** son idénticos por construcción: los tres
  brazos comparten el mismo bucle y el mismo punto de aplicación.
- El **coste de adquisición** que paga el receptor en el brazo S se informa
  **por separado** y nunca se absorbe en Φ.

> Un estudio que no informa su igualado no estima ρ.

## 6. Las tres manipulaciones y sus predicciones

| # | Regla manipulada | A por auditoría | Predicción |
|---|---|---|---|
| 0 | transferencia entre organismos, condición `E/p` | 1 | `δ < 0` ⇒ `ρ = NA` |
| 1 | proveedor de recurso reproductivo | 1 | `δ > 0`, `ρ` creciente con el tamaño del linaje competidor |
| 2 | supresión de competidores | 1 | `δ > 0`, `ρ ≈ 1` en todo el rango |

La manipulación 0 es el **control negativo**: la auditoría dice que la regla es
parcial, y sin embargo la dependencia basal no es positiva. Si el método
devolviera un `ρ` pequeño en vez de `NA`, estaría fabricando una medida donde
no hay dependencia que fraccionar. Es el brazo que comprueba que el método no
devuelve siempre lo que se espera.

## 7. Dosis-respuesta

`ρ` se estima para `n0_L1 ∈ {0, 40, 100, 200}` sobre `n0 = 400`.

Predicción específica y falsable: en `n0_L1 = 0` la regla parcial **no tiene a
quién desfavorecer**, la dependencia basal se anula y `ρ = NA` en las tres
manipulaciones. Es el caso límite de H4 y la demostración empírica de que
*parcialidad de una regla ≠ dependencia del suministro parcial*.

## 8. Qué falsaría cada hipótesis

- **H1** queda falsada si alguna intervención responde a las dos coordenadas:
  si retirar el operador externo cambia algo en `C = I`, o si el reetiquetado
  cambia la asignación en `A = 0`.
- **H4** queda falsada si dos condiciones difieren en monocultivo con la misma
  semilla.
- La manipulación 2 queda falsada si `ρ` sale claramente por debajo de 1 con un
  igualado dentro de tolerancia: significaría que existe un sustituto
  exchangeable de la exención, que es justo lo que se afirma imposible.
- El conjunto queda invalidado si el control negativo (manipulación 0) devuelve
  un `ρ` distinto de `NA` con dependencia basal no positiva.

---

## 9. Desviaciones respecto a lo pre-registrado

Se registran aquí porque un pre-registro del que se desvía uno en silencio no
sirve para nada. Todas se decidieron sobre **pilotos**, antes de mirar los
resultados confirmatorios.

### 9.1. La manipulación 2 se informa como rango, no como punto igualado

**Lo pre-registrado** (§5): igualar el brazo S al brazo B en «fracción vetada»,
tolerancia ±5 %.

**Lo que se hizo**: barrer la intensidad del sustituto sobre
`suppress_frac ∈ {0,05; 0,15; 0,30; 0,45; 0,60}` e informar `ρ` en todo el
rango.

**Por qué**: la cantidad a igualar resultó inservible, por dos razones que se
descubrieron al calibrar.

1. El contador de eventos vetados cuenta **organismo-pasos bloqueados, no
   eventos impedidos**. En el brazo parcial los miembros del linaje no focal
   quedan vetados permanentemente y se recuentan en cada paso, mientras que en
   el brazo abierto el vetado de un paso puede no serlo en el siguiente. Los
   dos brazos se inflan con factores distintos y el cociente deja de ser
   comparable entre ellos.
2. Cambiado el objetivo a reproducción total realizada —que sí cuenta eventos
   una sola vez—, la bisección converge con error del 0,02 al 0,10 %, pero
   sobre una cantidad **casi insensible a la manipulación**: medido, los
   nacimientos totales se mueven entre 148.690 y 153.008 mientras
   `suppress_frac` recorre de 0,002 a 0,45. El mundo compensa la reproducción
   impedida. Fracciones que difieren en dos órdenes de magnitud «casan» igual
   de bien, y elegir entre ellas sería arbitrario.

Igualar sobre una cantidad insensible no es igualar. La salida honesta es
informar `ρ` en todo el rango del sustituto: si no depende de él, la conclusión
es robusta frente a una elección de igualado que no sabemos hacer de forma
única. Es una afirmación más débil que la pre-registrada y más defendible.

El igualado del **proveedor** (manipulación 1) no cambia: el volumen emitido es
directamente controlable, se iguala con error del 0,0 %, y el coste de
adquisición se informa por separado.

### 9.2. La predicción del control negativo queda falsada en dosis alta

**Lo pre-registrado** (§6): la manipulación 0 daría `ρ = NA` en todo el rango,
por dependencia basal no positiva.

**Lo medido**: `NA` en `n_L1 ∈ {0, 40}`, pero `δ > 0` y `ρ > 1` en
`n_L1 ∈ {100, 200}`.

**Por qué**: la predicción era incorrecta, no la medida. Una regla de asignación
parcial en población mixta no es solo «un apoyo que quizá no hacía falta»: es
además un **mecanismo de supresión de competidores**, porque el linaje no focal
aporta al fondo común y no es servido. Con competidor grande, retirar la regla
libera al competidor y perjudica al linaje focal, luego la dependencia basal se
vuelve positiva. Las manipulaciones 0 y 2 convergen en dosis alta.

Se informa como falsación de la predicción, que es para lo que existe el
pre-registro.

### 9.3. `ρ > 1` no estaba contemplado

El manuscrito describía `ρ` como «la fracción de una dependencia demostrada que
sobrevive a des-personalizar la regla», lo que invita a leerlo dentro de
`[0, 1]`. No está acotado por arriba, y aquí se miden valores por encima de 1:
significan `Φ_S < Φ_R`, es decir que des-personalizar la regla cuesta **más**
que retirarla. Es un estado distinguible y real —el sustituto abierto no deja
de favorecer al linaje focal, sino que además abastece a los rivales que la
regla excluía— y se ha añadido al manuscrito en §3.5.

### 9.4. Ventana de medida del acceso

No estaba especificada. Se fija en **600 pasos** tras la inyección, sobre
pilotos: con ventana larga los estados de clones y residentes divergen y una
regla que asigna por necesidad reparte distinto por razones de estado, no de
etiqueta (el cociente cae a 0,74–0,98 en vez de 1,00). Se exige además un
mínimo de **500 asignaciones** en la ventana para que el cociente sea
interpretable.

### 9.5. Criterio de la prueba de deriva

**Lo pre-registrado**: `N_e/N` aproximadamente constante al variar `N`, dentro
del 25–30 %.

**Lo que se aplicó**: `N_e` crece monótonamente con `N`, y hay pérdida de
varianza en las tres poblaciones. Se informa además el exponente del escalado.

**Ambos resultados, para que se pueda juzgar la relajación**:

| `N` censal | `N_e` estimado | `N_e/N` |
|---|---|---|
| 369 | 228 | 0,619 |
| 736 | 343 | 0,466 |
| 1475 | 495 | 0,336 |

Con el criterio original la prueba **falla**: la dispersión de `N_e/N` es del
59,7 %. Con el nuevo, pasa, y el escalado medido es `N_e ~ N^0,56`.

**Por qué el criterio original era el equivocado**: la expresión de Wright
supone generaciones discretas y descendencia poissoniana, y este mundo no tiene
ninguna de las dos —hay solapamiento generacional y mortalidad dependiente de
condición—. Exigir `N_e/N` constante era exigir al sustrato una propiedad que
el modelo no predice. Lo que la teoría sí exige, y es lo que se contrasta, es
que haya deriva y que su escala de tiempo crezca con el tamaño poblacional.

**Esto es una relajación de criterio en una prueba bloqueante, y como tal se
registra.** Queda pendiente de explicación mecanicista en el análisis C2: si la
varianza en número de descendientes crece con `N`, la caída de `N_e/N` deja de
ser una anomalía tolerada y pasa a ser una predicción cumplida.

### 9.6. Criterio de la prueba de insensibilidad, y el umbral de degeneración

**Lo pre-registrado**: el patrón se mantiene en todas las variantes de la
rejilla, incluida la exigencia de que la retirada del operador cambie el
resultado en `C = E`.

**Lo que se aplicó**: se exigen las **identidades estructurales** y se
**informa** el tamaño de efecto sin exigirlo.

- `C = I` → cociente `Φ_R/Φ_B` igual a 1,000000 exacto (tolerancia 1e−12)
- `A = 0` → cociente de acceso dentro del 5 % de 1
- `A = 1` → asignación cruzada exactamente 0

**Por qué**: exigir que la retirada siempre muerda es exigir que una regla
extrínseca siempre importe, que es justo lo que la desigualdad entre propiedad
de una regla y dependencia de un sistema niega. Donde no muerde, la salida
correcta es `δ ≈ 0` y `ρ = NA`, que es un resultado y no un fallo. Mantener el
criterio original habría obligado a declarar roto el sustrato en un régimen
donde el marco predice exactamente lo observado.

**Umbral numérico de «la regla no se dispara»**, declarado: una variante es
degenerada cuando se producen **menos de 500 asignaciones** en la ventana de
medida del acceso, promediadas sobre las semillas de esa variante
(`MIN_EVENTOS = 500` en `validacion.py`).

Se fijó sobre el diagnóstico de `diag_theta.py`, que midió el número de
asignaciones y el intervalo del cociente de acceso a cuatro umbrales
reproductivos:

| `theta` | asignaciones | cociente de acceso | IC 95 % | cubre 1 |
|---|---|---|---|---|
| 7,0 | 84 | 0,938 | [0,862, 1,018] | sí |
| 8,5 | 1.950 | 1,002 | [0,988, 1,017] | sí |
| 10,0 | 3.340 | 0,993 | [0,980, 1,005] | sí |
| 14,0 | 4.954 | 0,991 | [0,981, 1,000] | no |

El salto es de 84 a 1.950: no hay ninguna variante en la zona ambigua, así que
el umbral no está ajustado al caso que quería excluir.

**El criterio se evalúa para las once variantes por igual.** Recuento de
asignaciones en la ventana, medido y publicado para todas:

| variante | asignaciones | veredicto |
|---|---|---|
| por defecto | 3.193 | evaluada |
| `theta=7,0` | **74** | **degenerada** |
| `theta=14,0` | 4.925 | evaluada |
| `ks=0,01` | 3.775 | evaluada |
| `ks=0,04` | 2.675 | evaluada |
| `E_don=6,0` | 4.711 | evaluada |
| `E_don=9,0` | 1.627 | evaluada |
| `q=2,0` | 2.487 | evaluada |
| `q=4,0` | 3.937 | evaluada |
| `m_sample=4` | 3.159 | evaluada |
| `m_sample=16` | 3.512 | evaluada |

La variante degenerada está **22 veces** por debajo de la siguiente más baja.
El umbral de 500 cae en un intervalo vacío: ninguna decisión depende de dónde
se ponga exactamente dentro de [75, 1.600].
