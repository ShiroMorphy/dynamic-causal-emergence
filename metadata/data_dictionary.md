# EIA-930 Data Dictionary and Provenance Specification

## 1. Primary Data Source
- **Provider**: U.S. Energy Information Administration (EIA)
- **Form**: Form EIA-930 (Hourly Electric Grid Monitor)
- **API Version**: EIA API v2 (`https://api.eia.gov/v2/electricity/rto/`)
- **Reporting Frequency**: Hourly
- **Timestamp Standard**: Coordinated Universal Time (UTC)

## 2. API Endpoints & Variable Specifications

| Variable | API Endpoint Route | Field Name | Units | Description |
|---|---|---|---|---|
| Demand ($D_i$) | `region-data/data/` | `value` (type: `D`) | Megawatthours (MWh) | Hourly demand / load |
| Day-Ahead Forecast ($DF_i$) | `region-data/data/` | `value` (type: `DF`) | Megawatthours (MWh) | Day-ahead demand forecast |
| Net Generation ($NG_i$) | `region-data/data/` | `value` (type: `NG`) | Megawatthours (MWh) | Total net generation |
| Net Interchange ($NI_i$) | `region-data/data/` | `value` (type: `TI`) | Megawatthours (MWh) | Total net interchange (exports - imports) |
| Wind Generation ($W_i$) | `fuel-type-data/data/` | `value` (fuel: `WND`) | Megawatthours (MWh) | Wind turbine net generation |
| Solar Generation ($S_i$) | `fuel-type-data/data/` | `value` (fuel: `SUN`) | Megawatthours (MWh) | Utility-scale solar net generation |
| Coal Generation | `fuel-type-data/data/` | `value` (fuel: `COL`) | Megawatthours (MWh) | Coal thermal generation |
| Natural Gas Gen. | `fuel-type-data/data/` | `value` (fuel: `NG`) | Megawatthours (MWh) | Natural gas generation |
| Nuclear Generation | `fuel-type-data/data/` | `value` (fuel: `NUC`) | Megawatthours (MWh) | Nuclear baseload generation |
| Hydro Generation | `fuel-type-data/data/` | `value` (fuel: `WAT`) | Megawatthours (MWh) | Conventional hydro generation |

> [!NOTE]
> Frequency deviations ($\Delta f_i$) are NOT collected or reported by Form EIA-930 and are excluded from the dataset specification.

## 3. Physical & Operational Diagnoses

### Balancing Accounting Residual
$$Residual_{i,t} = NG_{i,t} + NI_{i,t} - D_{i,t}$$
Quantifies transmission losses, reporting discrepancies, and pumping/storage adjustments.

### Variable Renewable Energy (VRE) Penetration
$$VRE_t = \frac{\sum_{i \in \text{Region}} (W_{i,t} + S_{i,t})}{\sum_{i \in \text{Region}} NG_{i,t}}$$
Bounded between 0 and 1.

### Normalized Demand Forecast Error
$$FE_t = \frac{|D_t - DF_t|}{D_t}$$
Operational target for out-of-sample predictability evaluations.
