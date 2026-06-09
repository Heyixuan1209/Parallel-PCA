#include "pca.h"
#include "matrix.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <lapacke.h>

static int copy_first_k_columns(const double *src, size_t rows, size_t cols, int k, matrix_t **out) {
	int kk = (k <= 0 || k > (int)cols) ? (int)cols : k;
	matrix_t *dst = matrix_create(rows, (size_t)kk);
	if (!dst) return -1;
	for (int col = 0; col < kk; ++col) {
		memcpy(dst->data + (size_t)col * rows,
			   src + (size_t)col * rows,
			   rows * sizeof(double));
	}
	*out = dst;
	return 0;
}

int pca_by_eig_serial(const matrix_t *X, int k, double *eigvals, matrix_t **eigvecs, matrix_t **projection) {
	if (!X || !eigvals || !eigvecs || !projection) return -1;
	size_t n = X->rows;
	size_t d = X->cols;
	if (n < 2) return -1;
	int kk = (k <= 0 || k > (int)d) ? (int)d : k;
	matrix_t *C = matrix_create(d, d);
	if (!C) return -1;
	matrix_local_covariance(X, C);
	double scale = 1.0 / (double)(n - 1);
	for (size_t i = 0; i < d * d; ++i) C->data[i] *= scale;

	double *Vbuf = (double *)malloc(d * d * sizeof(double));
	if (!Vbuf) { matrix_free(C); return -1; }
	int rc = jacobi_eigen_serial(C->data, d, eigvals, Vbuf, 1e-10, (int)(100 * d + 10));
	if (rc != 0) { free(Vbuf); matrix_free(C); return rc; }

	*eigvecs = matrix_create(d, d);
	if (!*eigvecs) { free(Vbuf); matrix_free(C); return -1; }
	memcpy((*eigvecs)->data, Vbuf, d * d * sizeof(double));

	matrix_t *Vmat = NULL;
	if (copy_first_k_columns(Vbuf, d, d, k, &Vmat) != 0) {
		free(Vbuf);
		matrix_free(C);
		matrix_free(*eigvecs);
		*eigvecs = NULL;
		return -1;
	}
	*projection = matrix_create(n, kk);
	if (!*projection) {
		free(Vbuf);
		matrix_free(C);
		matrix_free(Vmat);
		matrix_free(*eigvecs);
		*eigvecs = NULL;
		return -1;
	}
	matrix_dgemm(1.0, X, Vmat, 0.0, *projection);

	free(Vbuf);
	matrix_free(C);
	matrix_free(Vmat);
	return 0;
}

int pca_by_eig_parallel_gather(const matrix_t *X_local, int k, MPI_Comm comm, int root, double *eigvals, matrix_t **eigvecs, matrix_t **projection_local) {
	if (!X_local || !eigvals || !eigvecs || !projection_local) return -1;
	int rank;
	MPI_Comm_rank(comm, &rank);
	size_t d = X_local->cols;
	matrix_t *C_local = matrix_create(d, d);
	if (!C_local) return -1;
	matrix_local_covariance(X_local, C_local);

	long long local_n = (long long)X_local->rows;
	long long total_n = 0;
	MPI_Allreduce(&local_n, &total_n, 1, MPI_LONG_LONG, MPI_SUM, comm);
	if (total_n < 2) { matrix_free(C_local); return -1; }
	double scale = 1.0 / (double)(total_n - 1);
	for (size_t i = 0; i < d * d; ++i) C_local->data[i] *= scale;

	double *Vbuf = (double *)malloc(d * d * sizeof(double));
	if (!Vbuf) { matrix_free(C_local); return -1; }
	int rc = jacobi_eigen_parallel_gather(C_local->data, d, eigvals, Vbuf, comm, root, 1e-10, (int)(100 * d + 10));
	if (rc != 0) { free(Vbuf); matrix_free(C_local); return rc; }

	*eigvecs = matrix_create(d, d);
	if (!*eigvecs) { free(Vbuf); matrix_free(C_local); return -1; }
	memcpy((*eigvecs)->data, Vbuf, d * d * sizeof(double));

	int kk = (k <= 0 || k > (int)d) ? (int)d : k;
	matrix_t *Vmat = NULL;
	if (copy_first_k_columns(Vbuf, d, d, k, &Vmat) != 0) {
		free(Vbuf);
		matrix_free(C_local);
		matrix_free(*eigvecs);
		*eigvecs = NULL;
		return -1;
	}

	*projection_local = matrix_create(X_local->rows, kk);
	if (!*projection_local) {
		free(Vbuf);
		matrix_free(C_local);
		matrix_free(Vmat);
		matrix_free(*eigvecs);
		*eigvecs = NULL;
		return -1;
	}
	matrix_dgemm(1.0, X_local, Vmat, 0.0, *projection_local);

	free(Vbuf);
	matrix_free(C_local);
	matrix_free(Vmat);
	return 0;
}

int pca_by_svd_serial(const matrix_t *X, int k, double *singvals, matrix_t **V, matrix_t **projection) {
	if (!X || !singvals || !V || !projection) return -1;
	lapack_int m = (lapack_int)X->rows;
	lapack_int n = (lapack_int)X->cols;
	if (m <= 0 || n <= 0) return -1;
	lapack_int minmn = (m < n) ? m : n;

	double *a = (double *)malloc((size_t)m * (size_t)n * sizeof(double));
	if (!a) return -1;
	memcpy(a, X->data, (size_t)m * (size_t)n * sizeof(double));

	double *vt = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
	if (!vt) { free(a); return -1; }
	double *s = (double *)calloc((size_t)minmn, sizeof(double));
	if (!s) { free(a); free(vt); return -1; }
	double *superb = (minmn > 1) ? (double *)malloc((size_t)(minmn - 1) * sizeof(double)) : NULL;

	int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'A', m, n, a, m, s, NULL, 1, vt, n, superb);
	if (info != 0) {
		free(a); free(vt); free(s); free(superb);
		return info;
	}

	*V = matrix_create(n, n);
	if (!*V) { free(a); free(vt); free(s); free(superb); return -1; }
	for (lapack_int i = 0; i < n; ++i) for (lapack_int j = 0; j < n; ++j) (*V)->data[i + j * n] = vt[i * n + j];

	int kk = (k <= 0 || k > (int)n) ? (int)n : k;
	*projection = matrix_create(X->rows, kk);
	if (!*projection) { free(a); free(vt); free(s); free(superb); matrix_free(*V); *V = NULL; return -1; }

	matrix_t *Vmat = NULL;
	if (copy_first_k_columns((*V)->data, (size_t)n, (size_t)n, k, &Vmat) != 0) {
		free(a); free(vt); free(s); free(superb); matrix_free(*V); *V = NULL; matrix_free(*projection); *projection = NULL; return -1;
	}
	matrix_dgemm(1.0, X, Vmat, 0.0, *projection);

	for (lapack_int i = 0; i < n; ++i) {
		singvals[i] = (i < minmn) ? s[i] : 0.0;
	}

	free(a); free(vt); free(s); free(superb); matrix_free(Vmat);
	return 0;
}

int pca_by_svd_parallel(const matrix_t *X_local, int k, MPI_Comm comm, int root, double *singvals, matrix_t **V_local, matrix_t **projection_local) {
	if (!X_local || !singvals || !V_local || !projection_local) return -1;
	int rank;
	MPI_Comm_rank(comm, &rank);
	size_t d = X_local->cols;
	int rc = 0;

	matrix_t *X_global = NULL;
	if (matrix_gather_rows_to_root(X_local, &X_global, root, comm) != 0) return -1;

	double *Vbuf = (double *)malloc(d * d * sizeof(double));
	if (!Vbuf) {
		if (rank == root) matrix_free(X_global);
		return -1;
	}

	if (rank == root) {
		lapack_int m = (lapack_int)X_global->rows;
		lapack_int n = (lapack_int)X_global->cols;
		lapack_int minmn = (m < n) ? m : n;
		double *a = X_global->data;
		double *vt = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
		double *s = (double *)calloc((size_t)minmn, sizeof(double));
		double *superb = (minmn > 1) ? (double *)malloc((size_t)(minmn - 1) * sizeof(double)) : NULL;
		if (!vt || !s) {
			free(vt);
			free(s);
			free(superb);
			rc = -1;
		}

		if (rc == 0) {
			int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'A', m, n, a, m, s, NULL, 1, vt, n, superb);
			if (info != 0) {
				rc = info;
			}
		}

		if (rc == 0) {
			for (lapack_int i = 0; i < n; ++i) {
				singvals[i] = (i < minmn) ? s[i] : 0.0;
				for (lapack_int j = 0; j < n; ++j) {
					Vbuf[i + j * n] = vt[i * n + j];
				}
			}
		}

		free(vt);
		free(s);
		free(superb);
		matrix_free(X_global);
	}

	MPI_Bcast(&rc, 1, MPI_INT, root, comm);
	if (rc != 0) {
		free(Vbuf);
		return rc;
	}

	MPI_Bcast(singvals, (int)d, MPI_DOUBLE, root, comm);
	MPI_Bcast(Vbuf, (int)(d * d), MPI_DOUBLE, root, comm);

	*V_local = matrix_create(d, d);
	if (!*V_local) { free(Vbuf); return -1; }
	memcpy((*V_local)->data, Vbuf, d * d * sizeof(double));

	int kk = (k <= 0 || k > (int)d) ? (int)d : k;
	matrix_t *Vmat = NULL;
	if (copy_first_k_columns(Vbuf, d, d, k, &Vmat) != 0) {
		free(Vbuf);
		matrix_free(*V_local);
		*V_local = NULL;
		return -1;
	}

	*projection_local = matrix_create(X_local->rows, kk);
	if (!*projection_local) {
		free(Vbuf);
		matrix_free(*V_local);
		*V_local = NULL;
		matrix_free(Vmat);
		return -1;
	}
	matrix_dgemm(1.0, X_local, Vmat, 0.0, *projection_local);

	free(Vbuf);
	matrix_free(Vmat);
	return 0;
}
