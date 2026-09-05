# Dynamic Causal Emergence under Nonstationarity
## Plan maestro de implementación, validación y publicación Q1

**Título de trabajo recomendado**

> **Dynamic Causal Emergence under Nonstationary Dynamics: Time-Local Estimation, Uncertainty, and Evidence from U.S. Power Grids**

**Nombre corto del método:** DCE  
**Estimador neuronal:** Dyn-NIS+  
**Aplicación principal:** EIA-930 / U.S. power grids  
**Objetivo editorial:** paper metodológico de sistemas complejos, información causal y energía, con código abierto y reproducibilidad integral.  
**Estado de este documento:** protocolo de investigación, implementación y auditoría.  
**Fecha de corte:** 2026-09-05.

---

# 0. Resumen ejecutivo

La idea tiene potencial Q1, pero para alcanzar ese estándar el paper debe transformarse desde una **demostración conceptual con resultados prototipo** a una contribución metodológica con cuatro pilares verificables:

1. **Definición matemáticamente coherente del estimando**: qué significa exactamente \(EI_t\), qué intervención se está realizando, cómo se compara micro versus macro y bajo qué supuestos de no estacionariedad local.
2. **Estimador estadísticamente válido**: no basta una red que produzca una trayectoria visualmente plausible. Debe existir consistencia conceptual con causal emergence, control de sesgo, incertidumbre y selección de \(q_t^*\).
3. **Validación sintética con ground truth real**: varios Data Generating Processes (DGPs), cientos de réplicas Monte Carlo, nulos difíciles, cambios abruptos y suaves, variación de ruido y comparación con métodos contemporáneos.
4. **Aplicación empírica auténtica y reproducible**: EIA-930 real, sin datos generados, sin estadísticas simuladas para construir p-values o figuras, sin variables que EIA-930 no contiene y sin leakage temporal.

El paper no debe intentar “demostrar” de antemano que mayor VRE implica mayor DCE ni que existe un “phase transition”. Esas deben ser **preguntas empíricas falsables**.

La contribución Q1 debe formularse de manera más precisa:

> **DCE no es simplemente calcular CE en ventanas móviles.** El aporte debe ser un framework que estima una familia temporal de mecanismos \(P_t\), coarse-grainings \(\phi_t\), causal scales \(q_t^*\) e información efectiva interventional \(EI_t\), con regularización temporal, operación one-sided para inferencia online, calibración de incertidumbre y tests explícitos para cambios de escala causal.

---

# 1. Auditoría crítica del estado actual

## 1.1 Dictamen

**Conservar:**
- pregunta científica;
- idea de \(DCE_t\);
- coexistencia de estimador local y estimador neuronal;
- dimensión causal dinámica \(q_t^*\);
- benchmark sintético + power grid;
- intención de comparación multiescala;
- orientación open-source.

**Rehacer antes de usar cualquier número como resultado científico:**
- estimador de Effective Information;
- pipeline empírico EIA-930;
- surrogate testing;
- análisis de Winter Storm Uri;
- forecast evaluation;
- claims sobre phase transition;
- CE 2.0;
- tablas de benchmarks;
- cifras cuantitativas del abstract.

**Regla:** todas las cifras actuales deben tratarse como **mockups / prototype figures** hasta ser regeneradas desde datos reales y pipelines auditables.

---

# 2. Reposicionamiento científico de la novedad

## 2.1 Claim que NO conviene hacer

No usar:
> “This is the first method to compute causal emergence over time.”
Ya existen trabajos que calculan medidas de causal emergence mediante sliding windows en aplicaciones.

## 2.2 Claim defendible si la implementación cumple el protocolo

> We introduce a framework for **time-indexed causal emergence under nonstationary transition mechanisms** that jointly estimates locally evolving transition laws and coarse-graining maps, provides one-sided online inference, intervention-consistent effective information, uncertainty-calibrated dynamic causal dimensionality, and explicit detection of changes in causal scale.

---

# 3. Preguntas de investigación definitivas

## RQ1 — Methodological
Can causal emergence be estimated reliably when the underlying transition mechanism and optimal coarse-graining vary over time?

## RQ2 — Statistical
Can DCE distinguish genuine changes in causal scale from nonstationary noise, heteroskedasticity, seasonal drift and ordinary correlation changes?

## RQ3 — Power systems
Does the causal organization of U.S. power-system dynamics vary systematically with operating conditions, renewable penetration and extreme stress?

## RQ4 — Operational value
Does past-only DCE contain incremental out-of-sample information for forecast errors or operational stress beyond standard dynamic, spectral and network measures?

---

# 4. Hipótesis reformuladas para evitar overclaiming

- **H1 (Presence)**: Inferencia sobre episodios, FDR, nulo IAAFT re-estimado.
- **H2 (VRE modulation)**: Regresión no lineal con controles de demanda, hora, estacionalidad y clima.
- **H3 (Extreme events)**: Event study emparejado con controles para múltiples eventos (Uri, California heatwaves, etc.).
- **H4 (Predictability)**: Evaluación walk-forward estrictamente causal sin leakage.

---

# 5. Formalización matemática que debe quedar cerrada antes de programar

## 5.1 Proceso no estacionario y suavidad local
$$d(P_{t+u}, P_t) \le L |u|^\alpha$$

## 5.2 Kernels temporales
- Retrospectivo: $s \in [t-h, t+h]$
- Causal (one-sided): $s \le t$ ($w_{t,s} = 0 \ \forall s > t$)

## 5.3 Medida de intervención $\nu_t^{(d)}$
- Opción A: $\mathcal{U}([-a, a]^d)$ con $\Sigma_{do} = \frac{a^2}{3} I_d$
- Opción B: $\mathcal{N}(0, I_d)$ con $\Sigma_{do} = I_d$

## 5.4 Dynamic Effective Information
$$EI_t(Z) = I_{\nu_t}(do(Z_t); Z_{t+1}) = H_{\nu_t}(Z_{t+1}) - \mathbb{E}_{Z_t \sim \nu_t}[H(P_t(Z_{t+1} \mid do(Z_t)))]$$

## 5.5 Benchmark exacto lineal gaussiano
$$EI_t(X) = \frac{1}{2} \ln \frac{\det(A_t \Sigma_{do} A_t^\top + \Sigma_{\epsilon, t})}{\det(\Sigma_{\epsilon, t})}$$

## 5.6 Macroproyección y DCE
$$DCE_t(q) = EI_t(V^{(q)}) - EI_t(X)$$
$$DCE_t^{\text{norm}}(q) = \frac{EI_t(V^{(q)})}{q} - \frac{EI_t(X)}{p}$$

---

# 6. Secuencia óptima de ejecución

$$\boxed{ \text{Theory (M1)} \rightarrow \text{Exact estimator (M2)} \rightarrow \text{Calibration (M3)} \rightarrow \text{Dyn-NIS+ (M4)} \rightarrow \text{Real EIA (M5)} \rightarrow \text{Inference (M7)} \rightarrow \text{Forecast (M8)} \rightarrow \text{Release (M10)} }$$
