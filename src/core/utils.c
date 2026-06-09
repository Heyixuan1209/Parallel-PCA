#include "utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double pca_wtime(void) {
	int mpi_initialized = 0;
	MPI_Initialized(&mpi_initialized);
	if (mpi_initialized) {
		return MPI_Wtime();
	}

	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void pca_mpi_abort_if_error(int errcode, const char *msg, MPI_Comm comm) {
	if (errcode == 0) return;
	int rank = 0;
	MPI_Comm_rank(comm, &rank);
	if (rank == 0 && msg) {
		fprintf(stderr, "%s (err=%d)\n", msg, errcode);
	}
	MPI_Abort(comm, errcode);
}

double pca_frobenius_norm(const double *A, size_t rows, size_t cols) {
	if (!A) return -1.0;
	double sumsq = 0.0;
	size_t total = rows * cols;
	for (size_t idx = 0; idx < total; ++idx) {
		sumsq += A[idx] * A[idx];
	}
	return sqrt(sumsq);
}

int pca_compute_global_mean(const matrix_t *X_local, double *mu, MPI_Comm comm) {
	if (!X_local || !mu) return -1;

	size_t n_local = X_local->rows;
	size_t d = X_local->cols;
	double *local_sum = (double *)calloc(d, sizeof(double));
	if (!local_sum) return -1;

	for (size_t j = 0; j < d; ++j) {
		const double *col = X_local->data + j * n_local;
		for (size_t i = 0; i < n_local; ++i) {
			local_sum[j] += col[i];
		}
	}

	int mpi_initialized = 0;
	MPI_Initialized(&mpi_initialized);
	if (!mpi_initialized) {
		if (n_local == 0) {
			free(local_sum);
			return -1;
		}
		for (size_t j = 0; j < d; ++j) {
			mu[j] = local_sum[j] / (double)n_local;
		}
	} else {
		long long local_n_ll = (long long)n_local;
		long long global_n_ll = 0;
		MPI_Allreduce(&local_n_ll, &global_n_ll, 1, MPI_LONG_LONG, MPI_SUM, comm);
		if (global_n_ll <= 0) {
			free(local_sum);
			return -1;
		}

		MPI_Allreduce(local_sum, mu, (int)d, MPI_DOUBLE, MPI_SUM, comm);
		for (size_t j = 0; j < d; ++j) {
			mu[j] /= (double)global_n_ll;
		}
	}

	free(local_sum);
	return 0;
}

static double explained_ratio_from_variances(const double *vars, size_t d, int k) {
	if (!vars || d == 0) return 0.0;
	size_t kk = (k <= 0 || (size_t)k > d) ? d : (size_t)k;
	double top = 0.0;
	double total = 0.0;
	for (size_t i = 0; i < d; ++i) {
		double v = vars[i];
		if (v < 0.0) v = 0.0;
		total += v;
		if (i < kk) top += v;
	}
	return total > 0.0 ? top / total : 0.0;
}

double pca_explained_ratio_from_eigvals(const double *eigvals, size_t d, int k) {
	return explained_ratio_from_variances(eigvals, d, k);
}

double pca_explained_ratio_from_singvals(const double *singvals, size_t d, int k) {
	if (!singvals || d == 0) return 0.0;
	double *vars = (double *)malloc(d * sizeof(double));
	if (!vars) return 0.0;
	for (size_t i = 0; i < d; ++i) {
		double sigma = singvals[i];
		vars[i] = sigma > 0.0 ? sigma * sigma : 0.0;
	}
	double ratio = explained_ratio_from_variances(vars, d, k);
	free(vars);
	return ratio;
}

void pca_print_block(const double *A, size_t rows, size_t cols, size_t maxr, size_t maxc, const char *name) {
	if (!A) return;
	if (name) {
		printf("%s =\n", name);
	}
	size_t rr = rows < maxr ? rows : maxr;
	size_t cc = cols < maxc ? cols : maxc;
	for (size_t i = 0; i < rr; ++i) {
		for (size_t j = 0; j < cc; ++j) {
			printf("%12.5e ", A[j * rows + i]);
		}
		if (cc < cols) printf("...");
		printf("\n");
	}
	if (rr < rows) printf("...\n");
}
