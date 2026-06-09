#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p bin results data build

FORCE_REBUILD=${FORCE_REBUILD:-0}

if [ "$FORCE_REBUILD" = "1" ]; then
  echo "Forcing rebuild: removing previous CMake cache and binaries..."
  rm -rf build
  mkdir -p build
  rm -f bin/pca_eig_ser bin/pca_eig_par_pro bin/pca_eig_par_thr bin/pca_svd_ser bin/pca_svd_par
fi

if [ -f CMakeLists.txt ]; then
  echo "Building with CMake..."
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -- -j
else
  echo "CMakeLists.txt not found, fallback compile may need manual link flags."
  mpicc -O3 -Iinclude src/core/matrix.c src/core/jacobi_ser.c src/core/jacobi_par.c src/core/pca.c src/core/utils.c src/benchmark/eig_ser.c -o bin/pca_eig_ser -llapacke -llapack -lblas -lm || true
  mpicc -O3 -Iinclude src/core/matrix.c src/core/jacobi_ser.c src/core/jacobi_par.c src/core/pca.c src/core/utils.c src/benchmark/eig_par_pro.c -o bin/pca_eig_par_pro -llapacke -llapack -lblas -lm || true
  mpicc -O3 -fopenmp -Iinclude src/core/matrix.c src/core/jacobi_ser.c src/core/jacobi_par.c src/core/pca.c src/core/utils.c src/benchmark/eig_par_thr.c -o bin/pca_eig_par_thr -llapacke -llapack -lblas -lm || true
  mpicc -O3 -Iinclude src/core/matrix.c src/core/jacobi_ser.c src/core/jacobi_par.c src/core/pca.c src/core/utils.c src/benchmark/svd_ser.c -o bin/pca_svd_ser -llapacke -llapack -lblas -lm || true
  mpicc -O3 -Iinclude src/core/matrix.c src/core/jacobi_ser.c src/core/jacobi_par.c src/core/pca.c src/core/utils.c src/benchmark/svd_par.c -o bin/pca_svd_par -llapacke -llapack -lblas -lm || true
fi

SAMPLES=${SAMPLES:-2000}
FEATURES=${FEATURES:-200}
RANK=${RANK:-20}
NOISE=${NOISE:-0.01}
SEED=${SEED:-2026}
K_LIST=${K_LIST:-"5 10 20 40"}
PROCS=${PROCS:-"2 4"}
DATASET=${DATASET:-data/synthetic.csv}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
GOTO_NUM_THREADS=${GOTO_NUM_THREADS:-1}
OMP_PROC_BIND=${OMP_PROC_BIND:-true}
OMP_PLACES=${OMP_PLACES:-cores}
PCA_GATHER_PROJECTION=${PCA_GATHER_PROJECTION:-0}
PCA_EIG_PRO_MODES=${PCA_EIG_PRO_MODES:-"root_serial_jacobi mpi_jacobi"}

export OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS
export MKL_NUM_THREADS
export GOTO_NUM_THREADS
export OMP_PROC_BIND
export OMP_PLACES
export PCA_GATHER_PROJECTION

echo "Generating data: samples=$SAMPLES features=$FEATURES rank=$RANK noise=$NOISE seed=$SEED"
echo "Runtime threading: OMP_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "Process-level eig modes: $PCA_EIG_PRO_MODES"
python3 data/generate_data.py \
  --samples "$SAMPLES" \
  --features "$FEATURES" \
  --rank "$RANK" \
  --noise "$NOISE" \
  --seed "$SEED" \
  --out "$DATASET"

RESULTS=results/benchmarks.csv
echo "method,n,d,k,num_procs,t_read,t_center,t_pca,t_total,explained_ratio" > "$RESULTS"

for k in $K_LIST; do
  echo "Running serial eig with k=$k ..."
  ./bin/pca_eig_ser "$DATASET" "$k" "results/eig_ser_k${k}.csv" >> "$RESULTS"

  echo "Running serial svd with k=$k ..."
  ./bin/pca_svd_ser "$DATASET" "$k" "results/svd_ser_k${k}.csv" >> "$RESULTS"

  for p in $PROCS; do
    for eig_mode in $PCA_EIG_PRO_MODES; do
      echo "Running process-level eig with mode=$eig_mode, k=$k, procs=$p (OMP=$OMP_NUM_THREADS, BLAS=$OPENBLAS_NUM_THREADS) ..."
      mpirun -np "$p" ./bin/pca_eig_par_pro "$DATASET" "$k" "results/eig_par_${eig_mode}_k${k}" "$eig_mode" >> "$RESULTS"
    done

    echo "Running thread-level eig with k=$k, procs=$p (OMP=$OMP_NUM_THREADS, BLAS=$OPENBLAS_NUM_THREADS) ..."
    mpirun -np "$p" ./bin/pca_eig_par_thr "$DATASET" "$k" "results/eig_par_root_omp_jacobi_k${k}" >> "$RESULTS"

    echo "Running parallel svd with k=$k, procs=$p (OMP=$OMP_NUM_THREADS, BLAS=$OPENBLAS_NUM_THREADS) ..."
    mpirun -np "$p" ./bin/pca_svd_par "$DATASET" "$k" "results/svd_par_k${k}" >> "$RESULTS"
  done
done

echo "Plotting results..."
python3 scripts/plot.py "$RESULTS" results/benchmark_summary.png

echo "Benchmarks complete."
echo "  CSV: $RESULTS"
echo "  PNG: results/benchmark_summary.png"
