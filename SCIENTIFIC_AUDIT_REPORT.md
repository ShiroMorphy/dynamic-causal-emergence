# Auditoría científica adversarial independiente

**Proyecto:** *Dynamic Causal Emergence: Detecting Time-Varying Macroscopic Causality in Nonstationary Complex Systems*  
**Commit auditado:** `94ec151940823fe0d6fd0f7bdd842931a3904f02`  
**Fecha:** 2026-09-05  
**Veredicto:** **A. DO NOT SUBMIT**

## 1. Dictamen ejecutivo

El repositorio no es científicamente certificable en su estado actual. La auditoría encontró bloqueadores independientes en la definición de la quantity principal, el estimador neural, los benchmarks sintéticos, la inferencia H1–H4, el tratamiento de EIA-930, CE2, las figuras, los baselines y el pipeline de reproducción.

El hecho central es más severo que una discrepancia de implementación: la quantity que se optimiza y reporta como emergencia es `EI_macro/q - EI_micro/p`, no la definición de Causal Emergence `EI_macro - EI_micro`. En todos los 12 parquets empíricos auditados, la DCE raw fue positiva en **0%** de los timestamps. H1, además, tiene media empírica raw negativa (`-0.13094`) y obtiene el p-value mínimo solo porque los surrogates son aún más negativos. Por tanto, el repositorio no establece `DCE_t > 0` bajo la definición raw declarada.

El segundo hecho decisivo es de trazabilidad: el script de reproducción no ejecuta las estimaciones ni H1–H4; verifica que existan artefactos, redibuja figuras y compila LaTeX. Varias figuras y la Tabla 1 contienen números manuales. Un artefacto nulo almacenado contradice una reejecución del código actual y no tiene procedencia verificable.

No se hicieron correcciones científicas. Se añadió únicamente una suite separada de falsación en `audit_tests/`; seis de siete pruebas fallan actualmente.

## 2. Alcance, inventario y DAG real

Se recorrieron recursivamente los 91 archivos versionados y los artefactos locales relevantes. El repositorio contiene:

- código Python en `src/dce/`: EI/intervenciones/kernels, estimadores lineales/SVD/Dyn-NIS+/NIS+, surrogates e hipótesis, EIA-930, DGPs A–I, runners, baselines y visualización;
- seis módulos de tests bajo `tests/` (38 tests);
- cuatro ZIP EIA-930 locales, parquets procesados, 12 trayectorias DCE empíricas, nueve JSON sintéticos, cuatro JSON H1–H4 y un parquet CE2;
- manuscrito LaTeX/PDF y siete figuras PDF;
- `pyproject.toml`, `Makefile`, manifests y metadata;
- directorios vacíos de configuración y CI;
- ningún notebook, suplemento, lockfile ni workflow CI operativo.

```text
ZIP EIA-930
  -> parse_raw_eia930_balance_zip
  -> eia930_{period}_hourly.parquet
  -> build_power_grid_microstate
  -> run_grid_estimation
  -> {ercot,western,eastern}_dce_*.parquet
  -> run_hypotheses_testing (en la práctica, 2021)
  -> h1/h2/h3/h4 JSON
  -> generate_all_paper_figures.py
  -> PDFs
  -> paper/main.tex -> paper/main.pdf

DGPs A-I -> run_synthetic_mc / compare_estimators
          -> JSON parciales -> Figura 2 / Tabla 1

EIA 2021 -> MultiscaleCE2Apportioner (invocado manualmente; sin runner)
         -> ce2_apportioning_2021.parquet -> Figura 6
```

Ramas sin linaje completo:

- `results/empirical/ce2_apportioning_2021.parquet` no tiene productor/escritor comprometido;
- `results/synthetic/oracle_closed_form.json`, citado por el claim ledger, no existe;
- Tabla 1 no tiene generador;
- la mayor parte de Figura 2 se construye con constantes;
- métricas PCA, runtimes y varias FPR no tienen raw result;
- los resultados 2022H2 no alimentan hipótesis ni figuras;
- ningún artefacto incluye commit, input hashes, entorno y configuración completos.

## 3. Tabla maestra de hallazgos

`OPEN` significa que el defecto sigue presente. `REFUTED` significa que una afirmación concreta fue contradicha por una recomputación. `UNVERIFIED` no se interpreta como evidencia favorable.

| ID | SEVERITY | DOMAIN | FILE | LINE/FUNCTION | PROBLEM | WHY IT MATTERS | EXACT FIX | REQUIRED RE-RUN | STATUS |
|---|---|---|---|---|---|---|---|---|---|
| DEF-01 | BLOCKER | Teoría | `paper/main.tex` / `effective_info.py` | Eq. DCE; `compute_dce` | La definición raw y la normalizada son quantities distintas. Ejemplo diagonal: raw `-2.6512`, normalizada `+0.1653`, con `q*=1`. | La normalización puede crear “emergencia” en canales independientes por sesgo dimensional. | Elegir una estimand primaria; si se conserva EI/dim, renombrarla y derivar su interpretación, null y selección. | Todos los DGPs, datos, H1–H4, figuras y claims. | OPEN |
| CLAIM-01 | BLOCKER | Claim central | `results/empirical/*dce*.parquet`; H1 JSON | todos | DCE raw positiva en 0% de los 12 parquets; media H1 raw `-0.13094`. | No se demuestra `EI_macro > EI_micro`. | Replantear claim o estimador y repetir inferencia sobre DCE raw preespecificada. | Paper completo. | REFUTED |
| EI-01 | BLOCKER | Matemática | `dyn_nis.py` | 107–144 | El logdet de covarianza se usa para salidas no gaussianas del decoder. Para `tanh(50z)+N(0,.1²)`, devuelve 2.2911 nats aunque MI ≤ `ln 2=.6931`. | El objetivo neural puede sobreestimar EI sin límite informativo válido. | Usar un bound/estimador MI válido o restringir y verificar la familia affine-Gaussian. | Dyn-NIS+, DGP H/I y empirical. | OPEN |
| EI-02 | BLOCKER | Numérica | `effective_info.py` | 91–124 | Ridge se añade una vez al ruido y dos veces al output. Con `A=0, Sigma=0`, EI=`d ln2/2`, no cero. | Produce información causal espuria en el null singular. | Regularizar un modelo generativo coherente una sola vez o usar pseudodeterminantes/rank-aware limits. | Oráculos, nulls y todos los resultados lineales. | OPEN |
| MACRO-01 | BLOCKER | Matemática | `linear_gaussian.py` | 126–135 | `W'AW` no es una dinámica macro cerrada salvo invariancia/lumpability; además se proyectan `V_t` y `V_{t+1}` con el mismo `W_t`, aunque la teoría usa `phi_t` y `phi_{t+1}`. | La EI macro calculada no corresponde al proceso coarse-grained declarado. | Estimar directamente `P(V_{t+1}|do(V_t))` con mapas temporales y condiciones de cerradura explícitas. | Teoría, estimator y todo downstream. | OPEN |
| DYN-01 | BLOCKER | Leakage | `dyn_nis.py`; `static_nis_plus.py` | 293–328; 128–154 | En bordes causales, si hay <8 muestras, `active_mask=all`; regularización/gauge usa futuro. | El estimador “online” ve futuro. | Prohibir fallback futuro; retrasar estimación hasta ESS mínimo y computar todos los términos con pesos causales. | DCE causal y H4. | OPEN |
| DYN-02 | MAJOR | ML objective | `dyn_nis.py` | loss | Escalamiento `V'=cV`: EI 1.4059→4.0293 al usar `c=.01`; la loss mejora. | El objetivo puede ser explotado por escala, no por causalidad. | Intervención covariante al reparametrizado, whitening/gauge exacto y test de invariancia. | Dyn-NIS+ completo. | OPEN |
| LOSS-01 | MAJOR | Manuscrito/código | `main.tex`; `dyn_nis.py` | loss; 317–345 | Pesos/términos no coinciden: reconstruction 0.3, falta norma θ, `loss_f` compara outputs y gauge es covarianza latente, no `W'W`. | La implementación no realiza el método publicado. | Una especificación única, ecuación-test y ablations por término. | Neural benchmarks/claims. | OPEN |
| PROC-01 | MAJOR | Procrustes | `dyn_nis.py` | temporal regularizer | La orientación SVD básica es correcta (error `8.6e-14`), pero la dinámica no se transforma por conjugación; rotaciones equivalentes dan loss 1.7333. | Penaliza gauge, no cambio funcional. | Alinear function space completo, incluida la dinámica conjugada; gradient checks. | Ablation/chattering. | OPEN |
| Q-01 | MAJOR | Selección | estimadores | selección q | Se maximiza y evalúa q en el mismo holdout; confidence set usa gap fijo .05 sin cobertura. | Winner's curse y q-recovery optimista. | Outer validation o inferencia selectiva/bootstrap max-stat. | q*, H1–H4. | OPEN |
| Q-02 | MAJOR | Teoría/código | `main.tex`; estimadores | grid q | Paper exige `q<p`; código permite `q=p` y fuerza DCE=0. | Inserta una opción nula artificial y altera q*. | Alinear dominio y tratar el micro nivel separadamente. | Todos los q*. | OPEN |
| TIME-01 | MAJOR | Temporal | `linear_gaussian.py` | 153–184 | Los pesos futuros son exactamente cero, pero la transición `X_t→X_{t+1}` se etiqueta t y se usa en t. | Fuga de una hora en cualquier claim online/forecast. | Etiquetar por tiempo de disponibilidad y usar outcomes ≤t. | H4 y causal DCE. | OPEN |
| TIME-02 | MAJOR | Supuestos | paper/kernels | local stationarity | No se formula Hölder/local stationarity, ESS ni boundary bias. ESS causal h=48 al inicio: 1,2,6,24.96,48. | No hay base asintótica para localización/inferencia. | Declarar supuestos, ESS mínimo, bandwidth selection y teoría de borde. | Simulación e inferencia. | OPEN |
| INT-01 | MAJOR | Intervención | `effective_info.py` | uniform | Usa fórmula gaussiana por igualdad de covarianza. Cuadratura escalar: MI uniforme .77739 vs fórmula .80472. | Covarianza igual no implica entropía/MI igual. | Implementar integral/convolution exacta o declarar aproximación/bound. | Claims de equivalencia y EI. | OPEN |
| COORD-01 | MAJOR | Invariancia | `microstate.py`; EI | scaling | “Whitened coordinates” es solo z-score marginal; no whitening conjunto. | EI y CE2 dependen de unidades/correlación. | Whitening train-only con transformación de intervención coherente y tests de unidades. | Empirical/CE2/H4. | OPEN |
| EST-01 | MAJOR | Estimación | `effective_info.py` | local regression | No hay intercept ni local centering. Una transición `y=2x+5` deja error máximo 5. | Deriva de media se confunde con dinámica/noise. | Añadir intercept/local centering y grados de libertad efectivos. | DGP-B/D y datos. | OPEN |
| SVD-01 | MAJOR | Baseline | `linear_svd.py` | projection | SVD(A) sin centering no maximiza EI. Contraejemplo `A=.5I`, `Sigma=diag(10,.01)`: proyección default `-.808`, alternativa `+.808` normalizada. | Baseline/oráculo no es óptimo. | Resolver la optimización EI bajo restricciones o etiquetar heurística. | Baselines. | OPEN |
| CE2-01 | BLOCKER | CE2 | `multiscale_ce2.py` | 88–113 | No implementa Causal Emergence 2.0: define sufficiency/necessity y shares como diferencias positivas de EI/dim. | Uso incorrecto del formalismo de Hoel. | Renombrar “EI-density heuristic inspired by CE2” o implementar causal primitives y ΔCP a lo largo del path. | Sección/Fig. 6. | OPEN |
| CE2-02 | BLOCKER | CE2/invariancia | mismo | apportioning | Shares MW `[98.7316,.9775,.2908,0]%`; los mismos datos en GW o ×1000 dan `[100,0,0,0]%`. | El claim 98.7/1.0/0.3 es una unidad de medida, no estructura causal. | Intervenciones coherentes entre escalas y prueba de invariancia. | CE2 completo. | REFUTED |
| ST-01 | BLOCKER | Monte Carlo | `run_synthetic_mc.py` | 87–109 | FPR se programa a cero: detecta null por substring en nombres `dgp_a/b/f/g`, que nunca contienen “null”/“shock”. | El control Type-I reportado es inválido. | Metadato `is_null` explícito y criterio calibrado. | Null MC/Fig.2c. | REFUTED |
| ST-02 | BLOCKER | Artefactos | `mc_null_nonstationary.json` | metrics | JSON reporta ceros perfectos; reejecución DGP-B seed1000 T600: RMSE .35677, bias .32783, max DCE .658, accuracy 0. | El artefacto no proviene del código actual. | Borrar artefactos, guardar resultados por réplica, commit/config/seed/hash. | Monte Carlo limpio. | REFUTED |
| ST-03 | BLOCKER | Benchmarks | `synthetic.py` | 143, 211–213, 270–279, 442–443, 501–502 | Ground truths son asignados. DGP-C nominal .6931 vs exact norm .4501/raw ≈−2.1e−5; DGP-E también difiere; Kuramoto/I son proxies ad hoc. | Bias/RMSE/q-recovery se miden contra targets incorrectos. | Derivar desde `A,Sigma,W` o declarar proxy fenomenológico. | DGP A–I. | REFUTED |
| ST-04 | MAJOR | Nulls | `synthetic.py` | A/B/F/G | Nulls isotrópicos tienen q* empatado, no q*=p único; DGP-G tiene DCE norm teórica +.08377. | FPR y q-recovery carecen de ground truth válido. | Targets set-valued; rediseñar/reclasificar G. | Null MC. | OPEN |
| ST-05 | MAJOR | Monte Carlo | `run_synthetic_mc.py` | delay/defaults | Default retrospectivo; detecciones desde τ−20 cuentan como éxito, produciendo delay negativo. R=20; 0/20 implica IC 95% bilateral superior 16.84%, no control perfecto. | Anticipación usa futuro y la incertidumbre es enorme. | Causal-only, anticipación=falsa alarma, muchas replicaciones e IC exactos. | Delay/FPR/power/coverage. | OPEN |
| H1-01 | BLOCKER | Surrogates | `surrogates.py` | 90–117 | La fase compartida solo existe al inicio; luego cada canal cambia fase. ERCOT T2000,p9: cross-spectrum error 19.23%, covarianza 9.36%, max KS .303; marginal exacta 0/9. | El null declarado no se preserva. | Implementar algoritmo multivariado validado y tolerancias de rechazo. | H1 B suficiente. | REFUTED |
| H1-02 | BLOCKER | Inferencia | `hypothesis.py`; H1 JSON | 132–150 | p=.001 compara media negativa con surrogates más negativos; max p=.003996. | No prueba emergencia positiva. | Separar null `DCE≤0` del contraste surrogate y exigir signo/estimand preespecificados. | H1/texto/abstract. | REFUTED |
| H1-03 | MAJOR | Scope | `run_hypotheses_testing.py` | 34–66 | Solo ERCOT, primeras 2000 h, LocalLinearGaussian retrospectivo; no Dyn-NIS+, 2 años ni 3 interconexiones. | Generalización del paper excede la prueba. | Ejecutar panel×año×estimador o restringir claim. | H1. | OPEN |
| H2-01 | MAJOR | Modelo | `hypothesis.py` | 178–206 | Es `LinearGAM`, no GAMM; sin random effects/AR. Residuales lag1 .997/.998 y lag24 .890/.921. | SE/p-values/bandas no son válidos. | GAMM real o GAM descriptivo con dependencia modelada/bootstrap temporal. | H2/Fig.4. | REFUTED |
| H2-02 | BLOCKER | Breakpoint | `hypothesis.py` | 213–268 | Western elige borde p85; bootstrap subsamplea grid y no puede devolver el endpoint; CI excluye γ̂ por construcción. Split-half 39.74% vs14.56%. | Breakpoint 37.2% no está identificado. | Grid ampliado/continuo, bootstrap bajo null con bloques y rechazo de frontera. | H2. | REFUTED |
| H2-03 | MAJOR | Test | `hypothesis.py` | 265–268 | Falta factor `sqrt(c)` de la propia fórmula Davies; Western paper dice p<1e−4 pero JSON=.0022487. | Claim numérico y test no coinciden. | Implementar sup-Wald validado por bootstrap y sincronizar paper. | H2. | REFUTED |
| H2-04 | MAJOR | Figura/endogeneidad | H2/Fig.4 | grid | Curva se calcula en min–max 1.49–66.10% pero se grafica contra percentiles 6.90–57.07%; DCE y VRE comparten inputs. | Eje falso y asociación mecánica; mecanismo no identificado. | Guardar/evaluar el mismo X-grid; medir mecanismos o usar lenguaje asociativo. | H2/Fig.4. | OPEN |
| H3-01 | BLOCKER | Event study | `hypothesis.py` | 321–369 | p=1.77e−4 es z-test iid horario; collapse lag1=.9828 y solo 3 runs. Placebo por bloques 8d p≈.283. | Pseudorreplicación extrema. | Inferencia a nivel evento/bloque y múltiples eventos. | H3. | REFUTED |
| H3-02 | BLOCKER | Matching | mismo | controls | p no usa controles matched; 2/4 controles solapan Uri. Figura muestra Feb10–21 aunque título dice Feb12–19. | Baseline contaminado; no hay efecto causal identificado. | Controles disjuntos y matching real con buffers/pretrends. | H3/Fig.5. | REFUTED |
| H4-01 | BLOCKER | Forecast | `hypothesis.py` | 451–469 | `train_y=fe[h:origin+h]` incluye targets hasta `origin+h−1`. | Fuga futura en h>1. | Incluir solo pares con outcome ≤ origen. | H4 todos horizontes. | REFUTED |
| H4-02 | MAJOR | Forecast | mismo | 443–506 | `train_window` no limita; es expanding. HAC/DM no refleja separación 12h ni suavizado; no guarda forecasts/CIs. | P-values publicados no corresponden al experimento descrito. | Rolling explícito, artefactos por origen, HAC en escala efectiva y corrección múltiple. | H4. | OPEN |
| DATA-01 | MAJOR | EIA-930 | `balance.py` | 16–27, 71–77 | EIA Total Interchange usa +export/−import; código calcula `NG+TI−D`. En 2021, 80.04% satisface `|NG−TI−D|≤1`, solo .307% la fórmula del código. | QC físico y reconciliación tienen signo inverso. | `NG−TI−D` o conversión explícita a net imports; no sobrescribir silenciosamente. | QC y pipelines afectados. | REFUTED |
| DATA-02 | MAJOR | Parsing | `client.py` | 131–142 | Totales adjusted pero fuels raw; outlier BANC gas 3,296,026 MW frente a adjusted 1,065 MW. | Estados extendidos físicamente imposibles. | Prioridad adjusted→imputed→raw y flags de procedencia. | Procesamiento/fuel/CE2. | OPEN |
| DATA-03 | MAJOR | Missingness | `microstate.py` | 104–105 | `ffill().bfill().fillna(0)`; 2022 tiene 504 NG faltantes, todas FPL. | Bfill filtra futuro y cero inventa energía. | Máscaras QC, imputación causal, panel común y sensitivity. | DCE/H4. | OPEN |
| DATA-04 | BLOCKER | Coverage | runner/git | `all`; raw | “2021–2022” no se ejecuta: `all=2021+2022H2`; H1–H4 usan 2021. ZIP 2022H1 requerido está sin trackear. | Clon limpio no contiene la muestra declarada. | Descargar/verificar 2022H1 o versionarlo; ejecutar años completos. | Empirical completo. | REFUTED |
| DATA-05 | MAJOR | Mapping | `ba_registry.csv`; `microstate.py` | hierarchy | Solo 21/64 BA; “RTO” incluye WECC_Other/SERC/FRCC y “Grid” mezcla interconexiones asíncronas. | Jerarquía no es nested ni representa continental grid. | Taxonomía temporal documentada y cobertura explícita; retirar nivel nacional. | CE2/claims. | OPEN |
| FIG-01 | BLOCKER | Figura 2 | `generate_all_paper_figures.py` | 37–89 | Curvas, accuracies, FPR y una barra hard-coded; τ cambia 150/250/500 según artefacto/texto/figura. | Evidencia visual fabricada/desvinculada. | Construir cada punto desde resultados por réplica. | Fig.2/Table1/claims. | REFUTED |
| FIG-02 | BLOCKER | Figuras | script/`main.tex` | Figs.3–6 | Fig.3 solo ERCOT 2021 aunque caption declara tres redes/surrogates; Fig.4 no Western/profile likelihood; Fig.5 fechas/controles incorrectos; Fig.6 carece de segundo panel. | Captions afirman evidencia ausente. | Regenerar desde artefactos correctos o corregir captions. | Figs.3–6. | REFUTED |
| BL-01 | BLOCKER | Baselines | `main.tex`; compare scripts | Table 1 | Tabla manual; PCA/FPR/runtime sin pipeline; información, tuning y compute no equivalentes. | Comparación de estado del arte no auditable ni fair. | Benchmark long-form por seed con mismas splits/budgets; generar tabla. | Baselines/Table1/Fig7. | UNVERIFIED |
| REP-01 | BLOCKER | Reproducibilidad | `reproduce_all.py` | 70–193 | No corre experiments; con resultados ausentes falla. Ante raw faltante puede terminar exit 0 y anunciar VERIFIED. Subprocess de figuras hereda cwd. | No existe reproducción desde clon limpio. | Orquestador fail-fast con cwd, data, tests, experiments, tablas, figuras y hashes. | Clon limpio. | REFUTED |
| REP-02 | MAJOR | Entrypoints | `Makefile`; `run_all.py` | targets/import | Config/módulo inexistentes; `run_all` no importa y mezcla EIA sintético, DCE surrogate aleatoria y net load aleatorio. | Entry points no ejecutan la ciencia publicada. | Eliminar mocks del pipeline empírico y añadir smoke CI. | Reproducción. | OPEN |
| TEST-01 | MODERATE | Tests | `tests/`; `audit_tests/` | suite | 38 tests pasan; 6/7 tests adversariales fallan. Tests existentes verifican shapes/mean/std, no contratos científicos. | “Tests green” no valida el método. | Adoptar oráculos, invariancia, leakage, surrogate, null, missing/timezone y lineage tests. | Suite completa. | OPEN |

## 4. Auditoría ecuación → supuesto → código → test → resultado

| Ecuación/objeto | Supuesto necesario | Implementación | Test independiente | Resultado |
|---|---|---|---|---|
| `EI=I(do(X_t);X_{t+1})` lineal-gaussiana | Canal affine-Gaussian, intervención gaussiana especificada, covarianza SPD | logdet en `effective_info.py` | Sistemas gaussianos bien condicionados | PASS numérico, error ≈`1e-12` |
| misma fórmula con singularidad | límite consistente o pseudodeterminante | doble ridge | `A=0,Sigma=0,d=3` | FAIL: 1.03972077 vs 0; condición infinita |
| intervención uniforme | entropía de convolution uniforme+gaussiana | reutiliza máximo gaussiano por covarianza | cuadratura escalar | FAIL: abs .02733, rel 3.52% |
| `DCE=EI_q−EI_p` | misma intervención inducida y macro dinámica válida | `compute_dce(... normalized=False)` pero selección/reporting usa norm | diagonal independiente | FAIL científico: raw −2.6512, norm +.1653 |
| `q*=argmax DCE` | objetivo único y validación externa | argmax normalizado, incluye q=p | null/isotropic y holdout | FAIL: empates y winner's curse |
| kernel causal | transición completa disponible al timestamp | weights futuros cero; etiqueta anticipada | perturbación `X_{t+1}` | FAIL online por 1 h |
| Dyn-NIS+ EI | decoder affine-Gaussian o MI bound | cov-logdet de output no lineal | `tanh(50z)` | FAIL: 2.2911 > ln2 |
| Procrustes | transformación consistente de encoder/decoder/dynamics | alinea pesos, no conjuga dinámica | rotaciones equivalentes | orientación PASS; invariancia funcional FAIL (1.7333) |
| local stationarity | regularidad `d(P_{t+u},P_t)≤L|u|^α`, ESS | no declarada | ESS de borde | FAIL/UNVERIFIED |

## 5. Oráculos y benchmarks A–I

Los diez casos mínimos solicitados están cubiertos por oráculos directos y/o por los DGPs. La conclusión adversarial es que varios “ground truths” del repositorio no son ground truth de EI/DCE.

| Caso | Instancia | Resultado conocido/recalculado | Estado |
|---|---|---|---|
| identidad + ruido | `A=I`, `Sigma=σ²I` | fórmula gaussian exacta en SPD; errores ≈`1e-12` | PASS limitado |
| diagonal lineal | `diag(.9,.8,.7,.6)`, `.1I` | raw q1 −2.6512; norm +.1653 | demuestra sesgo dimensional |
| rank-deficient | `A=0,Sigma=0` | EI exacta 0; código d ln2/2 | FAIL |
| variables redundantes | DGP-C | nominal .6931; exact norm .4501, raw ≈−2.1e−5 | FAIL ground truth |
| no-emergence | DGP-A | DCE norm empatada en q; q*=p no único | FAIL q target |
| known-emergence | DGP-C | target declarado no derivado | UNVERIFIED/REFUTED |
| changing-q | DGP-E | nominal [.3466,.6931,1.0397], exact [.6441,.6476,.3729] | FAIL |
| heteroskedastic null | DGP-F | FPR code no reconoce el null | FAIL |
| correlation shock | DGP-G | DCE norm teórica +.08377 | no es null |
| smooth drift | DGP-D | `true_dce=rho*constant` asignada, no EI de `A_t,Sigma_t` | UNVERIFIED |

DGP-H (Kuramoto) asigna `max(0,(R−.3)*1.5)` y q por umbral `R>.6`; DGP-I asigna `.5 log(cluster_size)` y hace clipping. Ninguno constituye CE analítica. Parámetros/defaults y seeds están en `synthetic.py`; Monte Carlo usa R=20 por defecto y no guarda suficiente detalle por réplica. Bias, RMSE, power, coverage y q-recovery publicados deben considerarse **UNVERIFIED** hasta rehacer los targets y el protocolo. Para 0 errores en 20 runs, el límite superior bilateral exacto 95% es 16.84% (unilateral ≈13.9%), no 0% certificado.

## 6. Auditoría EIA-930 desde raw

Evidencia positiva: los cuatro ZIP locales verifican sus hashes; parser→parquet reproduce exactamente los archivos trackeados 2021 y 2022H2. No hay duplicados `(timestamp,BA)` ni saltos UTC dentro de cada BA. DST queda absorbido por UTC.

| Año | Filas | BA | Ventana UTC | Pares BA-hora ausentes en unión | Nulls relevantes |
|---|---:|---:|---|---:|---|
| 2021 | 183,960 | 21 | 2021-01-01 06:00 a 2022-01-01 08:00 | 63 | forecast 327; interchange 97 |
| 2022 | 183,960 | 21 | 2022-01-01 06:00 a 2023-01-01 08:00 | 63 | forecast 120; generation 504; interchange 50 |
| 2022H2 | 92,755 | 21 | parcial | 65 | forecast 3; interchange 2 |

Residual 2021: con fórmula del código `NG+TI−D`, media −860 MW, mediana −803, MAE 3,373, p95 abs 13,087; con convención EIA `NG−TI−D`, media −164, mediana 0, MAE 295, p95 abs 2,182 y 75.0% exactos. En 2022, MAE 3,426 vs 330 MW y 76.25% exactos con el signo correcto. La documentación oficial de EIA indica negativo para inflow y positivo para outflow: [EIA, Hourly Electric Grid Monitor](https://www.eia.gov/Todayinenergy/detail.php?id=63684).

La muestra contiene 21 BA, no todo el sistema continental. No se encontró predictor meteorológico; afirmaciones de thermal dispatch, renewable ramping e inter-area synchronization no están medidas directamente.

## 7. H1–H4

- **H1:** null surrogate no preservado; full refit sí ocurre en la rutina H1, pero sobre ERCOT T=2000 y estimador local, no Dyn-NIS+ continental. B almacenado=1000 y p-value usa corrección Monte Carlo, pero la interpretación de signo es inválida y la inferencia temporal no usa cluster/max-T adecuado.
- **H2:** asociación GAM, no GAMM. Breakpoints e intervalos no son válidos; Western está en frontera y el CI no puede incluirlo por la grilla de bootstrap. “Phase transition” y mecanismos físicos no están identificados.
- **H3:** Uri no demuestra efecto causal ni structural reorganization. No existe evidencia de prespecificación anterior a observar DCE; se requiere validación multi-evento. Fechas registry/código difieren y controles solapan evento.
- **H4:** no es zero-future-information. Hay leakage de target y de timestamp DCE; `train_window` no hace rolling. Los p-values publicados son **REFUTED** como walk-forward leak-free. La conclusión cualitativa nula permanece **UNVERIFIED**, no confirmada.

## 8. Claim ledger independiente

| CLAIM | SOURCE CODE | RAW RESULT | RECOMPUTED | MANUSCRIPT | STATUS |
|---|---|---|---|---|---|
| 98.7% q recovery | hard-coded Fig.2/Table1 | Dyn-NIS MC: 20.3% (5 runs) | single-seed post-shift 98.66%, full MC no | 98.7% | REFUTED |
| 80% chattering reduction | two JSON bars | ratio de un seed, `Var(diff(DCE))` | sin ablation Procrustes ni IC | 80% | UNVERIFIED |
| FPR 0% | MC bug/hard-code | 0/20 almacenado | A/B/F/G superan umbral; flag siempre false | 0% | REFUTED |
| H1 p=.001 | H1 JSON | mean empirical −.13094 | p mínimo vs null más negativo | CE positiva | REFUTED |
| ERCOT γ=21.6% | H2 JSON | punto reproducible | inferencia y eje inválidos | 21.6%, p<1e−4 | UNVERIFIED |
| Western γ=37.2% | H2 JSON | borde grid, p=.0022487 | split-half 39.74/14.56 | 37.2%, p<1e−4 | REFUTED |
| Uri 11.04→19.27%, p=1.77e−4 | H3 JSON | iid hourly z | block placebo p≈.283 | causal collapse | REFUTED |
| H4 p-values | H4 JSON | leaked expanding fit | corrected alignment cambia h24 .148→.058; rolling2000 p=.0146 y empeora RMSE 1.71% | null result | REFUTED/UNVERIFIED |
| CE2 shares 98.7/1.0/0.3 | CE2 parquet | reproducible en MW | en GW 100/0/0/0 | multiscale causal apportioning | REFUTED |
| GAM pseudo-R² 34.8/36.3 | H2 JSON | valores almacenados | dependencia residual no tratada | p<1e−15 | point values only; inference UNVERIFIED |

## 9. Figuras, tablas y revisión visual

Los siete PDF se renderizaron e inspeccionaron. No se detectó corrupción gráfica, pero sí discrepancias científicas visibles:

- Fig.2 presenta números/curvas no generados por los resultados;
- Fig.3 muestra solo ERCOT 2021 y tres paneles distintos de la caption “tres interconexiones/surrogate distribution”;
- Fig.4 muestra ERCOT, no el panel Western ni profile likelihood declarado, y usa eje VRE incorrecto;
- Fig.5 grafica Feb10–21 bajo título Feb12–19;
- Fig.6 se titula CE2 exacto y solo tiene un panel;
- Fig.7 refleja el JSON H4, pero ese experimento tiene leakage;
- Tabla 1 está insertada manualmente y cita un método SVD como “Static PCA”.

El manuscrito PDF compila y tiene 10 páginas, pero la corrección tipográfica no compensa la ausencia de linaje.

## 10. Reproducibilidad y tests

Resultados ejecutados:

- `pytest -q`: **38 passed**;
- import de `run_all`: falla por `compute_benchmark_metrics` inexistente;
- import de `network_metrics`: falla por `Tuple` no importado;
- reproducción en copia sin `results/`: falla al buscar `mc_delay_comparison.json`;
- clon/archivo de HEAD no contiene `eia930-2022half1.zip`;
- `reproduce_all.py` puede devolver 0 y anunciar VERIFIED pese a data failure;
- suite adversarial: **6 failed, 1 passed**.

Las seis fallas adversariales son: EI nula singular, rechazo de NaN/Inf, intervención uniforme, intercept affine, marginal IAAFT exacta y signo EIA interchange. La prueba cross-spectral sintética construida pasó su tolerancia; esto no rescata H1: el panel ERCOT real falló con 19.23% de error cross-spectral.

## 11. Orden mínimo de reparación

1. Congelar claims y retirar cualquier uso de “verified”, 98.7%, FPR=0, H1 positiva, Uri causal, CE2 exacto y 2021–2022 continental.
2. Resolver la estimand: raw CE versus EI por dimensión, intervención y dinámica macro inducida.
3. Corregir EI singular/no lineal, invariancias, temporal availability y loss Dyn-NIS+.
4. Rediseñar DGPs con oráculos verdaderos; reejecutar MC causal con R preespecificado e IC.
5. Reprocesar EIA completo con signo, adjusted fuels, missingness causal, cobertura y hierarchy correctos.
6. Rehacer H1–H4 con protocolos estadísticos válidos y múltiples eventos.
7. Implementar baselines fair y generar Tabla/Figuras únicamente desde artefactos trazables.
8. Crear reproducción fail-fast desde clon limpio y CI.

## 12. Certificación

No se emite matriz de certificación final porque las categorías C/D no se alcanzan. Toda propiedad no demostrada arriba queda **UNVERIFIED**. El único veredicto compatible con la evidencia es:

# A. DO NOT SUBMIT

Referencia formal CE2 usada para contrastar nomenclatura: [Hoel, *Causal Emergence 2.0: Quantifying emergent complexity*](https://arxiv.org/abs/2503.13395); versión publicada: [Patterns 7(1), 101472](https://www.sciencedirect.com/science/article/pii/S2666389925003204).
