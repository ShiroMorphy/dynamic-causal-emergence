#!/bin/bash
# Reproduccion limpia desde cero de Dynamic Causal Dimensionality (DCD) en el cluster uni.
# Autor: Felipe Mora-Rojas
#
# Uso desde tu Mac:
#   scp reproduce_on_uni.sh uni:~/
#   ssh uni 'bash ~/reproduce_on_uni.sh'
#
# El pipeline dura horas. Se lanza con nohup, asi que puedes cerrar la sesion ssh
# sin matarlo. Para seguirlo:  ssh uni 'tail -f ~/dynamic_causal_emergence/repro_clean.log'

set -euo pipefail

REPO=~/dynamic_causal_emergence
PY=/home/glaurung/miniconda3/envs/torch-gpu/bin/python
PYTEST=/home/glaurung/miniconda3/envs/torch-gpu/bin/pytest

cd "$REPO"

echo "=== 1. Sincronizar con origin/main ==="
git fetch origin
git checkout main
git pull origin main
git log --oneline -1

echo "=== 2. Suite de tests antes de reproducir (deben pasar 86) ==="
PYTHONPATH=src "$PYTEST" -q tests/ audit_tests/

echo "=== 3. Purga de artefactos: nada se puede reutilizar silenciosamente ==="
# Los datos crudos NO se tocan: se verifican por SHA-256 contra el manifiesto Zenodo.
rm -rf results/empirical results/synthetic paper/figures
rm -f paper/main.pdf paper_dce_submitted.pdf
mkdir -p results/empirical results/synthetic paper/figures
echo "empirical=$(ls results/empirical | wc -l) synthetic=$(ls results/synthetic | wc -l) figures=$(ls paper/figures | wc -l)"

echo "=== 4. Reproduccion completa desde cero + compilacion LaTeX ==="
# Un hilo BLAS por proceso: el pipeline paraleliza por procesos y el thrashing
# de OpenBLAS degrada fuerte el refit de surrogates.
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

nohup "$PY" -u scripts/reproduce_all.py --from-scratch --compile-latex \
  > "$REPO/repro_clean.log" 2>&1 &

echo "Lanzado con PID $!. Log: $REPO/repro_clean.log"
echo
echo "Al terminar, verificar:"
echo "  python3 -c \"import pypdf; print(len(pypdf.PdfReader('paper/main.pdf').pages))\"   # debe imprimir 12"
echo "  grep -E 'n_surrogates|\"T\"' results/empirical/h1_surrogate_results.json | head"
echo "  git status --short   # que artefactos cambiaron respecto al freeze"
