# Dynamic Causal Emergence: Detecting Time-Varying Macroscopic Causality in Nonstationary Complex Systems
## *Evidence from Renewable-Dominated Power Grids*
### Protocolo de Implementación Riguroso y Exhaustivo para Publicación Q1 (Nature Comms / PRX / Chaos / IEEE Trans)

---

## 1. Resumen Ejecutivo y Arquitectura Conceptual del Proyecto

### 1.1 El Gap Científico y la Novedad
La teoría clásica de **Causal Emergence (CE)** (Hoel et al., 2013; Klein & Hoel, 2020) y sus extensiones continuas basadas en redes neuronales como **Neural Information Slicing (NIS / NIS+)** (Rosenthal et al., 2022; Zhang et al., 2024) asumen implícitamente **estacionariedad**:
$$P(X_{t+1} \mid X_t) = P(X_{s+1} \mid X_s) \quad \forall t, s$$

En sistemas complejos del mundo real (redes eléctricas con alta penetración renovable, mercados financieros, redes ecológicas, clima y cerebro), los mecanismos subyacentes son **estocásticos y no estacionarios**. Un macroestado óptimo $\phi$ estimado de manera global promedia dinámicas incompatibles, enmascarando transiciones críticas, colapsos dimensionales transitorios y reorganizaciones causales locales.

**Contribución Principal de este Paper:**
1. **Fundamentación Teórica**: Formalización matemática del **Dynamic Causal Emergence ($DCE_t$)**, permitiendo proyecciones macroscópicas dependientes del tiempo $\phi_t$, dinámicas latentes variables $f_t$, y una dimensión causal óptima dinámica $q_t^*$.
2. **Método Algorítmico Dual**:
   - **Local Nonparametric Kernel CE**: Para regímenes continuos con kernels temporales $w_{t,s} = K\left(\frac{s-t}{h}\right)$.
   - **Dynamic Neural Information Slicing (Dyn-NIS+)**: Red neuronal variacional online con función de pérdida tridimensional: *Reconstrucción/Predicción + Maximizador de Información Efectiva ($EI_t$) + Regularizador de Consistencia Temporal $\mathcal{R}_{\text{temp}}$*.
3. **Validación Sintética Rigurosa**: Benchmarking sobre sistemas dinámicos con verdad terreno analítica (cambios de régimen, bifurcaciones de Hopf, osciladores de Kuramoto no estacionarios).
4. **Aplicación Empírica a Gran Escala**: Análisis del sistema eléctrico estadounidense (EIA-930: ~65 Balancing Authorities, millones de datos horarios 2018–2025+), demostrando la relación no lineal entre penetración renovable ($VRE_t$), colapso de dimensionalidad causal ($q_t^*$) y eventos de estrés severo (Winter Storm Uri, California Heatwaves, rampas eólicas).
5. **Causal Emergence 2.0**: Descomposición multiescala mediante *Causal Apportioning* (Balancing Authorities $\to$ Regional Transmission Organizations $\to$ Interconnections $\to$ US Grid).
6. **Librería de Código Abierto**: Paquete Python modular `dynamic-causal-emergence` (`dce`) con interfaz estándar Scikit-Learn / PyTorch, documentación Sphinx/MkDocs, tests automatizados y pipelines reproducibles con DVC/Make.

---

## 2. Formulación Matemática Rigurosa

```
+---------------------------------------------------------------------------------------------------+
|                                  ESPACIO MICROSCÓPICO (Dimensión p)                              |
|   X_t in R^p  ----( Dinámica Micro No Estacionaria: P_t(X_{t+1}|X_t) )---->  X_{t+1} in R^p      |
+---------------------------------------------------------------------------------------------------+
       |                                                                          |
       | Mapeo Macro Dinámico:                                                   | Mapeo Macro Dinámico:
       | V_t = \phi_t(X_t; \theta_t) in R^q                                     | V_{t+1} = \phi_t(X_{t+1}; \theta_t) in R^q
       v                                                                          v
+---------------------------------------------------------------------------------------------------+
|                                  ESPACIO MACROSCÓPICO (Dimensión q < p)                           |
|   V_t in R^q  ----( Dinámica Macro Local: f_t(V_t; \psi_t) )------------->  V_{t+1} in R^q       |
+---------------------------------------------------------------------------------------------------+
```

### 2.1 Microestado y Transición Temporal Local
Sea $X_t \in \mathcal{X} \subseteq \mathbb{R}^p$ el microestado del sistema en el instante $t$.
La dinámica del microestado sigue un proceso no estacionario gobernado por la densidad condicional dependiente del tiempo $P_t(X_{t+1} \mid X_t)$.

Para estimar localmente $P_t$, introducimos un kernel temporal simétrico $K_h(u) = \frac{1}{h} K(u/h)$ con ancho de banda $h > 0$:
$$w_{t, s} = \frac{K\left( \frac{s - t}{h} \right)}{\sum_{\tau=1}^T K\left( \frac{\tau - t}{h} \right)}$$

### 2.2 Información Efectiva Dinámica ($EI_t$)
Para un sistema continuo en dimensión $d$, la Información Efectiva bajo intervención uniforme sobre un soporte compacto $\Omega \subset \mathbb{R}^d$ se formula en términos de entropías diferenciales:
$$EI_t(X) = I_t(do(X_t \sim \mathcal{U}(\Omega)); X_{t+1}) = H_t(X_{t+1} \mid do(X_t \sim \mathcal{U}(\Omega))) - H_t(X_{t+1} \mid X_t)$$

- **Determinismo / Efectividad ($E_t$)**: $H_t(X_{t+1} \mid do(X_t \sim \mathcal{U}(\Omega)))$ mide la capacidad del estado actual para proyectar el sistema sobre un rango diverso de salidas futuras.
- **Ruido / Degeneración**: $H_t(X_{t+1} \mid X_t)$ penaliza la incertidumbre intrínseca o estocasticidad de las transiciones.

### 2.3 Mapeo Macroscópico Dinámico y Definición de $DCE_t$
Sea $\phi_t: \mathbb{R}^p \to \mathbb{R}^q$ ($q < p$) una proyección macroscópica parametrizada por $\theta_t$. El macroestado inducido es $V_t = \phi_t(X_t) \in \mathbb{R}^q$.

Definimos el **Dynamic Causal Emergence para una dimensión $q$**:
$$\boxed{DCE_t(q) = EI_t^{(q)}(V) - EI_t^{(p)}(X)}$$

Y la **Magnitud de Causal Emergence Dinámica y Dimensión Causal Óptima**:
$$\boxed{DCE_t = \max_{1 \le q < p} DCE_t(q)}$$
$$\boxed{q_t^* = \arg\max_{1 \le q < p} DCE_t(q)}$$

*Interpretación Física*: Si $DCE_t > 0$, la escala macroscópica de dimensión $q_t^*$ posee mayor poder predictivo-causal y menor indeterminación que el microestado completo en el instante $t$.

---

### 2.4 Arquitectura Neural: Dyn-NIS+ (Dynamic Neural Information Slicing)

El framework neuronal está compuesto por dos redes acopladas:
1. **Encoder Dinámico (Coarse-graining)**: $V_t = \phi(X_t; \theta_t) \in \mathbb{R}^q$.
2. **Dinámica Latente (Macro-predictor)**: $\hat{V}_{t+1} = f(V_t; \psi_t) \in \mathbb{R}^q$.
3. **Decoder / Reconstructor Auxiliar**: $\hat{X}_{t+1} = g(V_{t+1}; \omega_t) \in \mathbb{R}^p$.

```
                  +--------------------------+
                  |       X_t in R^p         |
                  +--------------------------+
                               |
                               | \phi(X_t; \theta_t)
                               v
                  +--------------------------+
                  |       V_t in R^q         |
                  +--------------------------+
                               |
                               | f(V_t; \psi_t)
                               v
+------------------+     +--------------------------+     +-------------------+
|  X_{t+1} (True)  |     |   \hat{V}_{t+1} in R^q   |     | Maximizar EI_t(V) |
+------------------+     +--------------------------+     +-------------------+
        ^                              |                            |
        | Reconstrucción               | g(\hat{V}_{t+1}; \omega_t) |
        +------------------------------+                            v
                                                      Loss = L_pred - \lambda EI_t + \eta L_temp
```

#### Función de Pérdida Global de Dyn-NIS+:
Para cada ventana temporal ponderada centrada en $t$:
$$\min_{\Theta_t = \{\theta_t, \psi_t, \omega_t\}} \mathcal{J}(\Theta_t) = \mathcal{L}_{\text{prediction}}(t) - \lambda \cdot \widehat{EI}_t(V) + \eta \cdot \mathcal{R}_{\text{temporal}}(\Theta_t, \Theta_{t-1}) + \gamma \cdot \mathcal{R}_{\text{ortho}}(\theta_t)$$

Donde:
1. **$\mathcal{L}_{\text{prediction}}(t)$ (Fidelidad Dinámica)**:
   $$\mathcal{L}_{\text{prediction}}(t) = \sum_{s=1}^T w_{t,s} \left[ \| \phi(X_{s+1}; \theta_t) - f(\phi(X_s; \theta_t); \psi_t) \|_2^2 + \alpha \| X_{s+1} - g(f(\phi(X_s; \theta_t); \psi_t); \omega_t) \|_2^2 \right]$$
2. **$\widehat{EI}_t(V)$ (Información Efectiva Estimada por MINE / InfoNCE / Estimador Gaussiano Local)**:
   Para dinámicas gaussianas locales:
   $$\widehat{EI}_t(V) = \frac{1}{2} \ln \frac{\det \Sigma_{do(V_{t+1})}}{\det \Sigma_{\text{residual}, t}}$$
3. **$\mathcal{R}_{\text{temporal}}$ (Consistencia y Suavidad Temporal)**:
   $$\mathcal{R}_{\text{temporal}}(\Theta_t, \Theta_{t-1}) = \|\theta_t - \theta_{t-1}\|_2^2 + \|\psi_t - \psi_{t-1}\|_2^2$$
4. **$\mathcal{R}_{\text{ortho}}$ (Regularización de No Colapso)**:
   Evita que el encoder colapse a dimensiones redundantes garantizando $\text{Cov}(V_t) \approx I_q$.

---

## 3. Estructura Completa del Repositorio de GitHub

El repositorio debe seguir el estándar de un paquete científico de producción (PEP 621, Poetry/Hatch, CI/CD, Tipado Estricto, Pruebas Unitarias al 90%+ de cobertura).

```
dynamic-causal-emergence/
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                 # Linting (Ruff), Type checking (Mypy), Pytest (Linux/macOS/Windows)
│   │   ├── docs.yml               # Publicación automática de documentación en GitHub Pages
│   │   └── publish.yml            # Publicación automática a PyPI con GitHub Releases
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
│
├── configs/                       # Configuraciones reproducibles (Hydra / YAML)
│   ├── experiment_synthetic.yaml  # Configuración para benchmarks analíticos
│   ├── experiment_eia930.yaml     # Configuración para dataset de red eléctrica
│   ├── model/
│   │   ├── dyn_nis.yaml           # Hiperparámetros de Dyn-NIS+
│   │   ├── kernel_dce.yaml        # Hiperparámetros de Kernel Estimator
│   │   └── svd_dce.yaml           # Hiperparámetros de SVD / Linear Dynamic CE
│   └── baselines/
│       ├── dpca.yaml
│       ├── dfm.yaml
│       └── t_vae.yaml
│
├── data/                          # Gestión de datos (Versionado con DVC)
│   ├── raw/                       # Archivos descargados EIA-930 (.csv, .json)
│   ├── interim/                   # Datos limpios con consistencia de balance
│   └── processed/                 # Matrices microestado X_t listas para experimentación
│
├── docs/                          # Documentación completa (Sphinx / MkDocs Material)
│   ├── index.md                   # Introducción, instalación y quickstart
│   ├── theory/
│   │   ├── mathematical_formulation.md
│   │   └── ce_2_0_apportioning.md
│   ├── api/                       # Referencia API autogenerada
│   └── tutorials/
│       ├── 01_synthetic_regimes.ipynb
│       └── 02_us_power_grid_casestudy.ipynb
│
├── notebooks/                     # Jupyter Notebooks de análisis y generación de figuras de paper
│   ├── 01_synthetic_validation.ipynb
│   ├── 02_eia930_data_curation.ipynb
│   ├── 03_dce_power_grid_main_results.ipynb
│   ├── 04_extreme_events_deep_dive.ipynb
│   ├── 05_baseline_comparisons.ipynb
│   └── 06_ce_2_0_multiscale_apportioning.ipynb
│
├── paper/                         # Código fuente del paper en LaTeX y figuras vectoriales
│   ├── main.tex
│   ├── references.bib
│   ├── sections/
│   │   ├── 01_introduction.tex
│   │   ├── 02_theory.tex
│   │   ├── 03_synthetic_results.tex
│   │   ├── 04_power_grid_empirical.tex
│   │   └── 05_discussion.tex
│   └── figures/                   # Figuras vectoriales exportadas (.pdf, .svg, .png 300dpi)
│
├── src/                           # Código fuente del paquete Python
│   └── dce/
│       ├── __init__.py            # Exposición de la API pública principal
│       ├── py.typed               # Marcador PEP 561 para compatibilidad Mypy
│       │
│       ├── core/                  # Módulos matemáticos fundamentales
│       │   ├── __init__.py
│       │   ├── entropy.py         # Estimadores de entropía continua (Kozachenko-Leonenko, Gaussiano, k-NN)
│       │   ├── effective_info.py  # Cálculo de EI, Determinismo y Degeneración
│       │   └── kernels.py         # Kernels temporales (Gaussiano, Epanechnikov, Tricube, Exponencial)
│       │
│       ├── estimators/            # Estimadores de Dynamic Causal Emergence
│       │   ├── __init__.py
│       │   ├── base.py            # Clase base abstracta BaseDynamicCE (Scikit-Learn API)
│       │   ├── local_kernel.py    # Estimador no paramétrico local con kernels
│       │   ├── dyn_nis.py         # Dynamic Neural Information Slicing (PyTorch)
│       │   ├── linear_svd.py      # Estimador analítico de baja complejidad basado en SVD dinámico
│       │   └── multiscale_ce2.py  # Causal Emergence 2.0 (Hoel 2026 Apportioning)
│       │
│       ├── baselines/             # Métodos comparativos competitivos
│       │   ├── __init__.py
│       │   ├── dynamic_pca.py     # PCA con ventanas móviles y pesos exponenciales
│       │   ├── dynamic_factor.py  # Dynamic Factor Models (State-Space / Kalman)
│       │   ├── temporal_vae.py    # Variational Autoencoder con regularización temporal
│       │   ├── complexity.py      # Permutation Entropy, Effective Rank, O-Information
│       │   └── network_metrics.py # Modularity dinámica, Spectral Gap, Centrality
│       │
│       ├── datasets/              # Conectores y generadores de datos
│       │   ├── __init__.py
│       │   ├── synthetic.py       # Generadores de regímenes sintéticos, Kuramoto, Lorenz acoplado
│       │   └── eia930/            # Pipeline de datos EIA-930
│       │       ├── __init__.py
│       │       ├── client.py      # Descargador concurrente con API EIA v2 y Bulk CSVs
│       │       ├── parser.py      # Estandarización de columnas y zonas horarias (UTC)
│       │       ├── balance.py     # Verificación de invariante físico (Gen + Imp = Dem + Exp)
│       │       └── microstate.py  # Constructor del vector de microestado multivariado
│       │
│       ├── stats/                 # Pruebas estadísticas y métodos de inferencia
│       │   ├── __init__.py
│       │   ├── surrogates.py      # Generador de surrogates IAAFT, VAR-surrogates, Block-Bootstrap
│       │   └── hypothesis.py      # Pruebas de hipótesis H1-H4, GAMs y modelos econométricos
│       │
│       ├── visualization/         # Módulo para generación de figuras estilo Nature/Science
│       │   ├── __init__.py
│       │   ├── trajectories.py    # Gráficos temporales de DCE_t y q_t^* con bandas de confianza
│       │   ├── grid_maps.py       # Mapas geoespaciales interactivos de BAs y macro-clusters
│       │   ├── regimes.py         # Diagramas de fase y transiciones de emergencia
│       │   └── style.py           # Configuración de Matplotlib/Seaborn para Q1 (paletas, tipografía)
│       │
│       └── utils/                 # Utilidades generales
│           ├── __init__.py
│           ├── logging.py         # Configuración de logs estructurados con Rich
│           └── validation.py      # Validadores de dimensiones, NaN y tensores
│
├── tests/                         # Suite completa de tests automatizados
│   ├── conftest.py                # Fixtures compartidas (tensores sintéticos, datos mock EIA)
│   ├── test_entropy.py
│   ├── test_effective_info.py
│   ├── test_local_kernel.py
│   ├── test_dyn_nis.py
│   ├── test_multiscale_ce2.py
│   ├── test_eia930_pipeline.py
│   ├── test_baselines.py
│   └── test_surrogates.py
│
├── .gitignore
├── .dvcignore
├── dvc.yaml                       # Pipeline reproducible de DVC (Data Version Control)
├── pyproject.toml                 # Configuración de empaquetado, dependencias y herramientas
├── Makefile                       # Automatización total de comandos de desarrollo y paper
├── README.md                      # Documentación principal para GitHub con badges
└── LICENSE                        # Licencia abierta (MIT / Apache 2.0)
```

---

## 4. Paso a Paso Detallado de Implementación (Fase por Fase)

```
+----------------------------------------------------------------------------------------------------+
|                                CRONOGRAMA DE EJECUCIÓN CIENTÍFICA                                  |
+----------------------------------------------------------------------------------------------------+
| FASE 1 | Setup del Entorno, Especificación Matemática y Arquitectura de Paquete (Días 1-3)        |
+--------+-------------------------------------------------------------------------------------------+
| FASE 2 | Implementación de Estimadores Numéricos y Red Neuronal Dyn-NIS+ (Días 4-8)               |
+--------+-------------------------------------------------------------------------------------------+
| FASE 3 | Benchmarking Sintético: Validación de Ground Truth y Métricas de Detección (Días 9-13)   |
+--------+-------------------------------------------------------------------------------------------+
| FASE 4 | Pipeline EIA-930: Ingesta, Curaduría y Construcción del Microestado (Días 14-19)          |
+--------+-------------------------------------------------------------------------------------------+
| FASE 5 | Experimentos Empíricos en la Red Eléctrica y Validación de Hipótesis H1-H4 (Días 20-26)   |
+--------+-------------------------------------------------------------------------------------------+
| FASE 6 | Baselines Comparativos, Causal Emergence 2.0 y Ablaciones (Días 27-31)                    |
+--------+-------------------------------------------------------------------------------------------+
| FASE 7 | Generación de Figuras Q1, Empaquetado PyPI y Redacción del Manuscrito (Días 32-38)        |
+----------------------------------------------------------------------------------------------------+
```

---

### FASE 1: Setup del Entorno, Arquitectura del Paquete y Configuración CI/CD

#### Paso 1.1: Inicialización del Repositorio y Entorno
- Crear repositorio en GitHub: `https://github.com/felipemorarojas/dynamic-causal-emergence`.
- Configurar entorno Python 3.10+ gestionado mediante `pyproject.toml` con `poetry` o `pip-tools`.
- Instalar dependencias base:
  - **Cálculo Numérico & ML**: `numpy`, `scipy`, `torch>=2.2`, `scikit-learn`, `numba`, `jax` (opcional para autograd ultra-rápido).
  - **Series de Tiempo & Redes**: `pandas`, `polars`, `networkx`, `statsmodels`, `pygam`.
  - **Datos & APIs**: `requests`, `aiohttp`, `tqdm`, `dvc`, `h5py`.
  - **Visualización Q1**: `matplotlib`, `seaborn`, `cartopy`, `cmocean`, `scienceplots`.
  - **Calidad de Código**: `ruff`, `mypy`, `pytest`, `pytest-cov`, `pre-commit`.

#### Paso 1.2: Configuración de `pyproject.toml`
Definir metadata formal para habilitar la instalación vía `pip install -e .`:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "dynamic-causal-emergence"
version = "0.1.0"
description = "Detecting Time-Varying Macroscopic Causality in Nonstationary Complex Systems"
authors = [{ name = "Felipe Mora-Rojas" }]
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
classifiers = [
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "torch>=2.1.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "polars>=0.20.0",
    "networkx>=3.0",
    "statsmodels>=0.14.0",
    "pygam>=0.9.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "mypy>=1.5.0",
    "ruff>=0.1.0",
    "pre-commit>=3.4.0",
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=1.3.0",
]
```

#### Paso 1.3: Automatización con `Makefile`
Crear un `Makefile` con directivas claras:
- `make setup`: Instala entorno virtual y pre-commit hooks.
- `make test`: Corre la suite completa de pruebas unitarias.
- `make lint`: Verifica formato y tipos con ruff y mypy.
- `make data`: Ejecuta descarga y procesamiento de datos EIA-930 vía DVC.
- `make figures`: Genera todas las figuras vectoriales del paper.
- `make paper`: Compila el PDF en LaTeX.

---

### FASE 2: Implementación de Estimadores Numéricos y Algoritmo Dyn-NIS+

#### Paso 2.1: Módulo Core de Información Efectiva Dinámica (`dce/core/`)
1. **`entropy.py`**:
   - `gaussian_differential_entropy(cov_matrix)`: Entropía diferencial analítica $H(X) = \frac{1}{2} \ln((2\pi e)^d \det \Sigma)$.
   - `knn_differential_entropy(X, k=5)`: Estimador no paramétrico de Kozachenko-Leonenko para densidades arbitrarias.
   - `conditional_entropy_gaussian(cov_joint, target_idx, condition_idx)`.
2. **`kernels.py`**:
   - Funciones de ponderación temporal: Gaussiano $K(u) = \exp(-u^2/2)$, Epanechnikov $K(u) = \frac{3}{4}(1-u^2)\mathbb{I}_{|u|\le 1}$, Tricube.
   - Algoritmo de selección de ancho de banda óptimo $h$ basado en Validación Cruzada Temporal (Time-Series Cross-Validation / Leave-Future-Out).
3. **`effective_info.py`**:
   - `compute_effective_information(P_cond, intervention_distribution='uniform')`.
   - `compute_effectiveness(EI, max_entropy)`.

#### Paso 2.2: Estimador No Paramétrico Local (`dce/estimators/local_kernel.py`)
- Clase `LocalKernelDCE(BaseDynamicCE)`:
  - Métodos: `fit(X)`, `transform(X)`, `fit_transform(X)`.
  - Pondera observaciones en una ventana móvil o kernel suave para estimar la matriz de covarianza local $\Sigma_t(X_{t+1}, X_t)$.
  - Calcula $EI_t(X)$ microscópico para cada $t$.
  - Evalúa todas las proyecciones candidatas (agrupamientos o combinaciones lineales) para encontrar $q_t^*$ y $DCE_t$.

#### Paso 2.3: Red Neuronal Dyn-NIS+ (`dce/estimators/dyn_nis.py`)
Implementar la arquitectura en PyTorch:
1. **Encoder `DynamicEncoder(nn.Module)`**:
   - Capas densas parametrizadas con MLP o convolución 1D con pesos $\theta$.
   - Salida: vector continuo $V_t \in \mathbb{R}^q$.
   - Normalización por capas (LayerNorm) o capa de ortogonalización de Gram-Schmidt diferenciable para prevenir colapso de representación.
2. **Predictor Latente `MacroDynamics(nn.Module)`**:
   - ResNet latente o MLP parametrizada con $\psi$ que predice $V_{t+1}$ a partir de $V_t$.
3. **Módulo de Información Efectiva Differentiable `DifferentialEILoss`**:
   - Estima $EI_t(V)$ utilizando el determinante regularizado de la covarianza de las predicciones latentes o mediante una red crítica auxiliar tipo MINE (Mutual Information Neural Estimation).
4. **Bucle de Entrenamiento Online / Sliding Window**:
   - Optimización con `torch.optim.AdamW`.
   - Para cada paso $t$, inicializa con $\Theta_{t-1}$ (warm-start) y optimiza la función de pérdida con regularizador $\|\Theta_t - \Theta_{t-1}\|_2^2$.
   - Guarda trayectorias de tensores: `emergence_history_`, `dim_history_`, `weights_history_`.

```python
# Ejemplo de API estándar para el usuario
from dce.estimators import DynamicNIS

model = DynamicNIS(
    macro_dims=[1, 2, 4, 8],
    window_size=168,       # 1 semana en datos horarios
    bandwidth=24.0,        # Ancho de banda temporal
    lambda_ei=1.5,         # Peso de Información Efectiva
    eta_temp=0.5,          # Regularización de suavidad temporal
    learning_rate=1e-3,
    device="cuda"
)

# Ajuste sobre matriz multivariada de microestado
model.fit(X_train)

# Extracción de trayectorias
dce_trajectory = model.emergence_             # Array 1D con DCE_t
optimal_dim = model.causal_dimension_         # Array 1D con q_t^*
macro_states = model.macro_representations_   # Dict de tensores V_t para cada q
```

---

### FASE 3: Benchmarking Sintético Riguroso (Zero-Ground-Truth Validation)

Para validar un paper metodológico en una revista Q1, es imperativo demostrar matemáticamente que el estimador detecta transiciones causales en sistemas sintéticos donde se conoce con exactitud el valor teórico de $CE$.

```
+---------------------------------------------------------------------------------------------------+
|                        BENCHMARK SINTÉTICO: CAMBIO DE RÉGIMEN CAUSAL                              |
+---------------------------------------------------------------------------------------------------+
|  RÉGIMEN 1: Micro-determinismo sin CE (t in [0, 1000))                                            |
|  - Transiciones 1 a 1 entre microestados.                                                         |
|  - Micro-EI = Macro-EI  ==>  CE = 0, DCE_t approx 0.                                              |
+---------------------------------------------------------------------------------------------------+
|  RÉGIMEN 2: Ruido Microscópico Degenerado + Coherencia Macroscópica (t in [1000, 2000))            |
|  - Los microestados individuales sufren de alta entropía condicional P(X_{t+1}|X_t).              |
|  - La suma o promedio de clusters macroscópicos sigue una dinámica estrictamente determinista.    |
|  - Micro-EI << Macro-EI  ==>  CE > 0, DCE_t > 0.                                                  |
+---------------------------------------------------------------------------------------------------+
```

#### Paso 3.1: Modelos Sintéticos de Referencia (`dce/datasets/synthetic.py`)
1. **Sistema de Cadenas de Markov con Cambio Estocástico**:
   - $p=8$ microestados organizados en 2 macroestados ($q=2$).
   - $t < 1000$: Matriz de transición $P_A$ (biyectiva a nivel micro, $CE = 0$).
   - $t \ge 1000$: Matriz de transición $P_B$ (ruidosa a nivel micro dentro de bloques, determinista entre bloques, $CE = 1.0 \text{ bit}$).
2. **Red de Osciladores de Kuramoto No Estacionarios**:
   - $N=32$ osciladores acoplados con acoplamiento dependiente del tiempo $K(t)$ y ruido intrínseco.
   - En acoplamiento crítico, surge sincronización por clusters donde la dinámica de fases colectivas posee mayor información efectiva que las fases individuales.
3. **Sistemas Dinámicos Caóticos Acoplados (Lorenz / Rössler Multiescala)**:
   - Microestados con fluctuaciones caóticas rápidas acoplados a un atractor macroscópico lento cuya fuerza de acoplamiento modula temporalmente.

#### Paso 3.2: Métricas de Rendimiento Cuantitativo
Evaluar el estimador frente a $M = 500$ realizaciones de Monte Carlo:
1. **Detection Delay ($\Delta \tau$)**:
   $$\Delta \tau = \tau_{\text{detected}} - \tau_{\text{true}}$$
   Donde $\tau_{\text{detected}} = \min \{ t \ge \tau_{\text{true}} : DCE_t > \mu_0 + 3\sigma_0 \}$.
2. **Tasa de Falsos Positivos (FPR)**:
   Proporción de instantes en el régimen nulo ($CE=0$) donde el algoritmo declara erróneamente $DCE_t > \text{umbral}$.
3. **Error Cuadrático Medio de Estimación de Emergencia (RMSE-CE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{T} \sum_{t=1}^T (\widehat{DCE}_t - DCE_t^{\text{true}})^2}$$
4. **Exactitud de Recuperación de Dimensionalidad Macroscópica**:
   $$\text{Acc}_q = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(\hat{q}_t^* = q_t^{*, \text{true}})$$

---

### FASE 4: Pipeline de Datos de la Red Eléctrica (EIA-930)

#### Paso 4.1: Ingesta y Descarga Automatizada (`dce/datasets/eia930/client.py`)
- Descargar datos horarios históricos de la API v2 de la EIA y Bulk Data Manifests para los **66 Balancing Authorities (BAs)** de EE.UU. desde 2018 hasta 2025+.
- Variables extraídas por cada Balancing Authority $i \in \{1, \dots, N\}$:
  1. **Demand ($D_{i,t}$)**: Carga eléctrica en MWh.
  2. **Demand Forecast ($\hat{D}_{i,t}$)**: Pronóstico operativo de carga a $t-24h$.
  3. **Total Generation ($G_{i,t}$)**: Generación neta total.
  4. **Wind Generation ($W_{i,t}$)**: Generación eólica.
  5. **Solar Generation ($S_{i,t}$)**: Generación solar (fotovoltaica y térmica).
  6. **Net Interchange ($I_{i,t}$)**: Importaciones netas menos exportaciones con BAs vecinos.
  7. **Otras fuentes**: Nuclear, Gas Natural, Carbón, Hidroeléctrica.

```
                  +----------------------------------------------+
                  |         EIA-930 Raw Bulk CSV / API v2        |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |      Time Zone Normalization (UTC Only)      |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |    Physical Balance & Interchange Sanity     |
                  |     (G_i + I_i \approx D_i) & (\sum I_ij = 0)|
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |    Multivariate Microstate Matrix X_t        |
                  |    [D_1, G_1, W_1, S_1, I_1, ..., D_N, ...]  |
                  |    Dim: T \approx 60,000 hrs x p \approx 330 |
                  +----------------------------------------------+
```

#### Paso 4.2: Limpieza, Control de Calidad y Validación de Invariantes Físicos (`balance.py`)
1. **Conversión Estricta a UTC**: Eliminar desfases horarios por cambios de estación (Daylight Saving Time).
2. **Invariante Físico de Balance de Potencia Local**:
   Para cada BA $i$ en cada instante $t$:
   $$\text{Residual}_{i,t} = |G_{i,t} + I_{i,t} - D_{i,t}|$$
   Si $\text{Residual}_{i,t} / D_{i,t} > 0.10$, marcar como anomalía y aplicar imputación espacialmente consistente.
3. **Consistencia de Red en Intercambios**:
   Para el grafo acoplado de BAs, la suma total de transferencias interregionales debe anularse:
   $$\sum_{i=1}^N I_{i,t} \approx 0$$
4. **Estandarización y Filtrado de Tendencias**:
   - Normalización z-score adaptativa con ventanas móviles de 30 días para evitar que los ciclos estacionales (invierno/verano) dominen espuriamente la matriz de covarianzas causales.

#### Paso 4.3: Construcción del Microestado Vectorial y Métricas Auxiliares
- **Vector de Microestado $X_t \in \mathbb{R}^p$**:
  $$X_t = [D_{1,t}, G_{1,t}, W_{1,t}, S_{1,t}, I_{1,t}, \dots, D_{N,t}, G_{N,t}, W_{N,t}, S_{N,t}, I_{N,t}]^T \quad (p \approx 66 \times 5 = 330)$$
- **Penetración de Energías Renovables Variables ($VRE_t$)**:
  $$VRE_t = \frac{\sum_{i=1}^N (W_{i,t} + S_{i,t})}{\sum_{i=1}^N G_{i,t}}$$
- **Error Porcentual de Pronóstico de Demanda del Sistema ($FE_t$)**:
  $$FE_t = \frac{\sum_{i=1}^N |D_{i,t} - \hat{D}_{i,t}|}{\sum_{i=1}^N D_{i,t}}$$

---

### FASE 5: Experimentos en la Red Eléctrica y Validación de Hipótesis Científicas

```
+----------------------------------------------------------------------------------------------------+
|                                MATRIZ DE HIPÓTESIS CIENTÍFICAS                                     |
+----+---------------------------------------------------+-------------------------------------------+
| ID | Hipótesis Teórica / Empírica                      | Protocolo Estadístico de Validación       |
+----+---------------------------------------------------+-------------------------------------------+
| H1 | Existencia de Causal Emergence en el Power Grid  | Surrogate Data Testing (IAAFT / VAR)      |
| H2 | Modulación no lineal de DCE por Penetración VRE  | Generalized Additive Models (GAMs)        |
| H3 | Colapso Dimensional Macroscópico en Crisis       | Extreme Event Epoch Analysis & q_t^*      |
| H4 | DCE como Predictor de Estrés e Inestabilidad     | Granger Causality & OOS Forecasting Eq.   |
+----+---------------------------------------------------+-------------------------------------------+
```

#### Paso 5.1: Validación de Hipótesis 1 (H1 - Existencia de Emergencia Causal Significativa)
- **Pregunta**: ¿Es $DCE_t > 0$ estadísticamente superior al azar o a correlaciones lineales espurias?
- **Protocolo**: Generar $B = 1000$ series surrogadas utilizando el algoritmo **IAAFT (Iterated Amplitude Adjusted Fourier Transform)** y surrogates VAR que preservan el espectro de potencias lineal y las distribuciones marginales pero destruyen el acoplamiento causal no lineal.
- **Cálculo de Significancia**:
  $$p\text{-value}_t = \frac{1}{B} \sum_{b=1}^B \mathbb{I}(DCE_t^{\text{surrogate}, b} \ge DCE_t^{\text{empirical}})$$

#### Paso 5.2: Validación de Hipótesis 2 (H2 - Reorganización Causal frente a Energías Renovables)
- **Pregunta**: ¿Cómo altera la variabilidad climática de eólica y solar la escala causal dominante de la red?
- **Protocolo**: Ajustar un Modelo Aditivo Generalizado (GAM):
  $$DCE_t = \alpha + f_1(VRE_t) + f_2(\text{Hour}_t) + f_3(\text{Season}_t) + f_4(\text{NetLoadRamp}_t) + \varepsilon_t$$
  Donde $f_k(\cdot)$ son splines de regresión penalizados (P-splines).
- **Hallazgo esperado**: Detección de puntos de inflexión (tipping points) donde $DCE_t$ exhibe transiciones de fase para umbrales críticos de $VRE_t$ (e.g., $VRE > 35\%$).

#### Paso 5.3: Validación de Hipótesis 3 (H3 - Colapso Dimensional durante Eventos Extremos)
Analizar en detalle estudios de caso históricos de estrés severo del sistema:
1. **Winter Storm Uri (Febrero 2021 - Texas / ERCOT)**: Congelamiento de fuentes térmicas y desconexiones masivas.
2. **Winter Storm Elliott (Diciembre 2022 - Costa Este)**: Caída simultánea de generación y picos históricos de demanda.
3. **Ola de Calor en California (Septiembre 2022 - CAISO)**: Estrés por rampas solares vespertinas (efecto curva de pato extremo).
4. **Análisis de Épocas**:
   - Comparar la trayectoria de $q_t^*$ y $DCE_t$ en periodos normales vs. ventanas de emergencia de 72 horas.
   - Demostrar que durante las crisis, $q_t^*$ colapsa (e.g., de $q^*=16$ a $q^*=3$), indicando que los subsistemas independientes pierden autonomía y el grid se comporta como un macro-bloque rígido y vulnerable.

#### Paso 5.4: Validación de Hipótesis 4 (H4 - Capacidad Predictiva de Inestabilidad y Error Operativo)
- **Ecuación Econométrica Predictiva**:
  $$FE_{t+h} = \beta_0 + \sum_{k=1}^K \beta_k FE_{t-k} + \sum_{j=0}^J \theta_j DCE_{t-j} + \sum_{j=0}^J \phi_j q_{t-j}^* + \gamma^T Z_t + \epsilon_{t+h}$$
  Donde $Z_t$ incluye controles estándar (temperatura, volatilidad de precios de energía, varianza de demanda).
- **Evaluación Out-of-Sample (OOS)**: Rolling forecast con ventanas deslizantes evaluando el incremento de $R^2$ OOS y reducción de RMSE.

---

### FASE 6: Baselines Comparativos, Ablaciones y Causal Emergence 2.0

```
+---------------------------------------------------------------------------------------------------+
|                        SUITE DE COMPARACIÓN CON METODOLOGÍAS RIVALES                              |
+------------------------------------+--------------------------------------------------------------+
| Categoría Metodológica             | Métodos de Referencia Implementados                          |
+------------------------------------+--------------------------------------------------------------+
| 1. Reducción de Dimensionalidad    | - Dynamic PCA (Rolling Window SVD)                           |
|                                    | - Dynamic Factor Models (DFM con filtro de Kalman)           |
|                                    | - Temporal Variational Autoencoder (t-VAE)                   |
|                                    | - Dynamic Mode Decomposition (DMD)                           |
+------------------------------------+--------------------------------------------------------------+
| 2. Medidas de Complejidad / Info   | - Permutation Entropy Dinámica (Bandt & Pompe)               |
|                                    | - Effective Rank Dinámico (Roy & Vetterli)                   |
|                                    | - O-Information Dinámica (Sinergia vs Redundancia de Rosas)  |
|                                    | - Dynamic Mutual Information Multivariada                    |
+------------------------------------+--------------------------------------------------------------+
| 3. Topología y Grafos de Red       | - Modularity Dinámica de Grafos de Flujo de Potencia         |
|                                    | - Spectral Gap del Laplaciano Dinámico                       |
|                                    | - Dynamic PageRank / Eigenvector Centrality                  |
+------------------------------------+--------------------------------------------------------------+
```

#### Paso 6.1: Benchmark Comparativo Sistemático (`dce/baselines/`)
- Demostrar que $DCE_t$ no es una simple reformulación de la correlación cruzada o de la varianza explicada por PCA.
- Matriz de correlación y pruebas de no redundancia (regresiones multivariadas) entre $DCE_t$ y todas las métricas baseline.

#### Paso 6.2: Implementación Rigurosa de Causal Emergence 2.0 (Hoel 2026) (`dce/estimators/multiscale_ce2.py`)
- Implementar el formalismo exacto de **Causal Apportioning**:
  En lugar de seleccionar únicamente la macroescala óptima $\arg\max$, descomponer la causalidad total del sistema en contribuciones atribuidas a múltiples niveles jerárquicos:
  $$\text{Causality}_{\text{total}}(t) = \mathcal{C}_{\text{Micro}}(t) + \mathcal{C}_{\text{Meso-BA}}(t) + \mathcal{C}_{\text{Regional-RTO}}(t) + \mathcal{C}_{\text{Interconnection}}(t)$$
- Jerarquía natural de la red eléctrica de EE.UU.:
  $$\text{Balancing Authorities (66)} \longrightarrow \text{RTO/ISO Regions (7)} \longrightarrow \text{Interconnections (3: East, West, ERCOT)} \longrightarrow \text{US Grid (1)}$$
- Calcular la necesidad causal ($\text{CN}$) y suficiencia causal ($\text{CS}$) de cada nivel a lo largo del tiempo.

---

### FASE 7: Generación de Figuras de Calidad Q1, Empaquetado y Publicación

#### Paso 7.1: Estándares Visuales para Revistas Q1 (`dce/visualization/style.py`)
- Formato: Exportación directa en `.pdf` vectorial y `.png` a 300+ DPI.
- Tipografía: Fuentes estándar Nature/Science (`Helvetica` / `DejaVu Sans`), tamaño de etiquetas 8–10pt.
- Paletas de Color: Mapas de color perceptualmente uniformes y accesibles para daltónicos (`viridis`, `magma`, `cmocean.balance`, `cmocean.phase`).

```
+---------------------------------------------------------------------------------------------------+
|                            CATÁLOGO DE FIGURAS PRINCIPALES DEL PAPER                              |
+----------+----------------------------------------------------------------------------------------+
| Figura 1 | Diagrama Conceptual: Marco Teórico de DCE, Dyn-NIS+ y Espacio Micro-Macro Dinámico     |
| Figura 2 | Validación Sintética: Detección de Cambios de Régimen, Delays y Recuperación de q*      |
| Figura 3 | Mapa Espaciotemporal y Trayectoria de DCE en la Red Eléctrica de EE.UU. (2018–2025)   |
| Figura 4 | Respuesta No Lineal: DCE vs Penetración Renovable (VRE) y Curvas de Transición de Fase |
| Figura 5 | Anatomía Causal de Eventos Extremos: Colapso de q* en Winter Storm Uri & CA Heatwaves |
| Figura 6 | Causal Emergence 2.0: Descomposición Jerárquica Multiescala (BA -> RTO -> Grid)        |
| Figura 7 | Comparativa de Rendimiento con 10 Baselines y Capacidad Predictiva Out-of-Sample       |
+----------+----------------------------------------------------------------------------------------+
```

#### Paso 7.2: Publicación del Paquete en PyPI y Pipeline Reproducible
1. **GitHub Releases & PyPI**:
   - Automatización mediante GitHub Actions en tags `v0.1.0`.
   - Empaquetado de ruedas binarias (wheels) puras y con optimizaciones C/Cython/Torch.
2. **Pipeline de Reproducibilidad en un solo comando**:
   - Crear `dvc.yaml` y `Makefile` de modo que un evaluador o revisor anónimo pueda clonar el repositorio y ejecutar:
     ```bash
     git clone https://github.com/felipemorarojas/dynamic-causal-emergence.git
     cd dynamic-causal-emergence
     make setup
     make run-experiments
     make build-figures
     ```
   - Almacenar datasets curados y pesos entrenados en Zenodo con asignación de DOI permanente.

---

## 5. Estrategia Editorial y Defensa ante Revisores Q1

### 5.1 Revistas Objetivo (Tier 1)
1. **Target Principal**:
   - *Nature Communications* (Sección: Applied Physics & Mathematics / Complex Systems).
   - *Physical Review X (PRX)* o *Physical Review Research* (American Physical Society).
2. **Targets Especializados de Alto Impacto**:
   - *Chaos: An Interdisciplinary Journal of Nonlinear Science* (AIP Publishing - Fast Track / Focus Issue).
   - *Applied Energy* (Elsevier - Impact Factor > 11, para enfoque fuertemente energético).
   - *IEEE Transactions on Network Science and Engineering* o *IEEE Transactions on Power Systems*.
   - *Patterns* (Cell Press - Metodologías de Ciencia de Datos y Complejidad).

### 5.2 Anticipación de Críticas y Respuestas a Revisores

| Crítica Probable del Revisor | Estrategia de Mitigación y Blindaje en el Manuscrito |
|---|---|
| **1. "El método es solo un PCA dinámico o un estimador de covarianza móvil."** | Demostrar formalmente que sistemas con covarianza estática idéntica pueden tener $EI$ radicalmente distinta debido a la no linealidad y asimetría de la intervención causal ($do(X)$). Incluir el benchmark contra Dynamic PCA en la Figura 7. |
| **2. "¿Cómo eligen el ancho de banda temporal $h$ sin hacer overfitting?"** | Documentar el protocolo formal de Validación Cruzada Temporal (Rolling-origin cross-validation) y presentar un análisis de sensibilidad exhaustivo de $DCE_t$ frente a variaciones de $h \in [6h, 168h]$ en el Supplementary Information. |
| **3. "¿Es el microestado propuesto físicamente representativo de la red?"** | Explicar que el vector $X_t$ incorpora las 5 variables termodinámicas y de potencia primarias ($D, G, W, S, I$) que determinan las ecuaciones de flujo de potencia cuasiestático en balance de área. |
| **4. "¿Cuál es la ventaja computacional sobre NIS+ original?"** | Dyn-NIS+ implementa *warm-starting* y optimización de segundo orden local amortizada, reduciendo el costo de re-entrenamiento de $O(T \cdot E)$ a $O(T \cdot k)$ donde $k \ll E$ iteraciones por ventana temporal. |
| **5. "¿Por qué no usar Transfer Entropy o Granger Causality convencional?"** | La causalidad de Wiener-Granger y Transfer Entropy miden *flujos dirigidos de información entre nodos en una misma escala*, mientras que Causal Emergence cuantifica *ganancia causal intrínseca a través de escalas espaciotemporales (Micro vs Macro)*. |

---

## 6. Lista de Verificación (Checklist) para Inicio Inmediato en GitHub

- [ ] **1. Inicialización Git**: Crear repo privado/público con `.gitignore` para Python, PyTorch y DVC.
- [ ] **2. Entorno y Dependencias**: Configurar `pyproject.toml` y verificar instalación limpia en entorno aislado (`conda` o `venv`).
- [ ] **3. CI/CD Pipeline**: Configurar `.github/workflows/ci.yml` con ejecución automática de `pytest` y `ruff`.
- [ ] **4. Core Module**: Implementar `dce/core/entropy.py` y `dce/core/kernels.py` con pruebas unitarias asociadas.
- [ ] **5. Synthetic Suite**: Construir los 3 benchmarks sintéticos y verificar la detección de $\Delta \tau$ en notebooks.
- [ ] **6. Ingesta EIA-930**: Ejecutar script de descarga asíncrona de datos de los 66 BAs y validar invariantes de balance en `data/processed/`.
- [ ] **7. Modelo Dyn-NIS+**: Programar la arquitectura PyTorch y validar la convergencia de pérdida de información efectiva.
- [ ] **8. Experimentos Empíricos**: Computar trayectorias de $DCE_t$ y $q_t^*$ sobre los 7 años de datos horarios de EE.UU.
- [ ] **9. Modelado Estadístico**: Ajustar GAMs para $DCE_t = f(VRE_t)$ y modelos predictivos para eventos de estrés.
- [ ] **10. Suite de Baselines**: Ejecutar los 10 algoritmos rivales sobre los mismos conjuntos de datos.
- [ ] **11. Renderizado de Figuras**: Generar las 7 figuras principales del paper con tipografía y paletas vectoriales.
- [ ] **12. Redacción y Documentación**: Redactar manuscrito en LaTeX y compilar documentación Sphinx/MkDocs para la librería.

---
*Documento generado con especificaciones técnicas completas para investigación y desarrollo de impacto Q1.*
