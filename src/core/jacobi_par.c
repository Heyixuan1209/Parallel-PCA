#include "jacobi.h"

#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static int g_jacobi_parallel_debug = 0;

void jacobi_parallel_set_debug(int debug_level) {
	g_jacobi_parallel_debug = debug_level;
}

static int build_row_partition(size_t n, int size, int *row_counts, int *row_displs, int *elem_counts, int *elem_displs) {
	size_t base = n / (size_t)size;
	size_t rem = n % (size_t)size;
	int row_offset = 0;
	int elem_offset = 0;

	for (int rank = 0; rank < size; ++rank) {
		size_t rows = base + ((size_t)rank < rem ? 1 : 0);
		size_t elems = rows * n;
		if (rows > (size_t)INT_MAX || elems > (size_t)INT_MAX) return -1;
		row_counts[rank] = (int)rows;
		row_displs[rank] = row_offset;
		elem_counts[rank] = (int)elems;
		elem_displs[rank] = elem_offset;
		row_offset += row_counts[rank];
		elem_offset += elem_counts[rank];
	}
	return 0;
}

static int owner_of_row(size_t row, size_t n, int size) {
	size_t base = n / (size_t)size;
	size_t rem = n % (size_t)size;

	if (base == 0) return (int)row;

	size_t cutoff = (base + 1) * rem;
	if (row < cutoff) return (int)(row / (base + 1));
	return (int)(rem + (row - cutoff) / base);
}

static void pack_global_colmajor_to_rowblocks(
	const double *global_colmajor,
	size_t n,
	const int *row_counts,
	const int *row_displs,
	int size,
	double *packed_rowmajor
) {
	for (int rank = 0; rank < size; ++rank) {
		size_t rows = (size_t)row_counts[rank];
		size_t row_start = (size_t)row_displs[rank];
		size_t offset = row_start * n;
		for (size_t local_row = 0; local_row < rows; ++local_row) {
			size_t global_row = row_start + local_row;
			for (size_t col = 0; col < n; ++col) {
				packed_rowmajor[offset + local_row * n + col] = global_colmajor[col * n + global_row];
			}
		}
	}
}

static void unpack_rowmajor_to_colmajor(const double *rowmajor, size_t n, double *colmajor) {
	for (size_t row = 0; row < n; ++row) {
		for (size_t col = 0; col < n; ++col) {
			colmajor[col * n + row] = rowmajor[row * n + col];
		}
	}
}

static void jacobi_sort_eigenpairs(double *eigvals, double *eigvecs, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		size_t best = i;
		for (size_t j = i + 1; j < n; ++j) {
			if (eigvals[j] > eigvals[best]) best = j;
		}
		if (best != i) {
			double tmp = eigvals[i];
			eigvals[i] = eigvals[best];
			eigvals[best] = tmp;
			if (eigvecs) {
				for (size_t row = 0; row < n; ++row) {
					double v = eigvecs[row + i * n];
					eigvecs[row + i * n] = eigvecs[row + best * n];
					eigvecs[row + best * n] = v;
				}
			}
		}
	}
}

static int jacobi_eigen_root_omp(double *A, size_t n, double *eigvals, double *eigvecs, double tol, int maxiter) {
	if (!A || !eigvals || !eigvecs || n == 0) return -1;

	for (size_t col = 0; col < n; ++col) {
		for (size_t row = 0; row < n; ++row) {
			eigvecs[row + col * n] = (row == col) ? 1.0 : 0.0;
		}
	}

	int converged = 0;
	for (int iter = 0; iter < maxiter; ++iter) {
		double max_off = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(max:max_off) schedule(static)
#endif
		for (long long p = 0; p < (long long)n; ++p) {
			for (size_t q = (size_t)p + 1; q < n; ++q) {
				double apq = fabs(A[q * n + (size_t)p]);
				if (apq > max_off) max_off = apq;
			}
		}

		if (g_jacobi_parallel_debug && (iter == 0 || iter % 10 == 0 || max_off <= tol)) {
			fprintf(stderr, "[jacobi_root_omp] sweep=%d max_off=%.6e tol=%.6e\n", iter, max_off, tol);
		}
		if (max_off <= tol) {
			converged = 1;
			break;
		}

		for (size_t p = 0; p < n - 1; ++p) {
			for (size_t q = p + 1; q < n; ++q) {
				double apq = A[q * n + p];
				if (fabs(apq) <= tol) continue;

				double app = A[p * n + p];
				double aqq = A[q * n + q];
				double tau = (aqq - app) / (2.0 * apq);
				double t = (tau >= 0.0)
					? 1.0 / (tau + sqrt(1.0 + tau * tau))
					: -1.0 / (-tau + sqrt(1.0 + tau * tau));
				double c = 1.0 / sqrt(1.0 + t * t);
				double s = t * c;
				double app_new = app - t * apq;
				double aqq_new = aqq + t * apq;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
				for (long long ii = 0; ii < (long long)n; ++ii) {
					size_t i = (size_t)ii;
					if (i == p || i == q) continue;
					double aip = A[p * n + i];
					double aiq = A[q * n + i];
					double new_ip = c * aip - s * aiq;
					double new_iq = s * aip + c * aiq;
					A[p * n + i] = new_ip;
					A[i * n + p] = new_ip;
					A[q * n + i] = new_iq;
					A[i * n + q] = new_iq;
				}

				A[p * n + p] = app_new;
				A[q * n + q] = aqq_new;
				A[q * n + p] = 0.0;
				A[p * n + q] = 0.0;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
				for (long long ii = 0; ii < (long long)n; ++ii) {
					size_t i = (size_t)ii;
					double vip = eigvecs[p * n + i];
					double viq = eigvecs[q * n + i];
					eigvecs[p * n + i] = c * vip - s * viq;
					eigvecs[q * n + i] = s * vip + c * viq;
				}
			}
		}
	}

	for (size_t i = 0; i < n; ++i) eigvals[i] = A[i * n + i];
	if (!converged && g_jacobi_parallel_debug) {
		fprintf(stderr, "[jacobi_root_omp] reached maxiter=%d before tol=%.6e\n", maxiter, tol);
	}
	jacobi_sort_eigenpairs(eigvals, eigvecs, n);
	return converged ? 0 : 1;
}

static int jacobi_eigen_parallel_distributed_root(
	double *A_global,
	size_t n,
	double *eigvals,
	double *eigvecs,
	MPI_Comm comm,
	int root,
	double tol,
	int maxiter
) {
	int rank = 0;
	int size = 1;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	int *row_counts = (int *)malloc((size_t)size * sizeof(int));
	int *row_displs = (int *)malloc((size_t)size * sizeof(int));
	int *elem_counts = (int *)malloc((size_t)size * sizeof(int));
	int *elem_displs = (int *)malloc((size_t)size * sizeof(int));
	double *packed_global = NULL;
	double *A_local = NULL;
	double *V_local = NULL;
	double *col_p_local = NULL;
	double *col_q_local = NULL;
	double *col_p_full = NULL;
	double *col_q_full = NULL;
	double *diag_local = NULL;
	double *diag_global = NULL;
	double *eigvecs_rowmajor = NULL;
	int local_ok = 1;
	int global_ok = 0;
	int converged = 0;

	if (!row_counts || !row_displs || !elem_counts || !elem_displs) local_ok = 0;
	if (local_ok && build_row_partition(n, size, row_counts, row_displs, elem_counts, elem_displs) != 0) local_ok = 0;

	int local_rows = local_ok ? row_counts[rank] : 0;
	int local_row_start = local_ok ? row_displs[rank] : 0;
	size_t local_rows_sz = (size_t)local_rows;

	if (local_ok && rank == root) {
		packed_global = (double *)malloc(n * n * sizeof(double));
		if (!packed_global) {
			local_ok = 0;
		} else {
			pack_global_colmajor_to_rowblocks(A_global, n, row_counts, row_displs, size, packed_global);
		}
	}

	if (local_ok) {
		A_local = (double *)malloc((local_rows_sz > 0 ? local_rows_sz * n : 1) * sizeof(double));
		V_local = (double *)malloc((local_rows_sz > 0 ? local_rows_sz * n : 1) * sizeof(double));
		col_p_local = (double *)malloc((local_rows_sz > 0 ? local_rows_sz : 1) * sizeof(double));
		col_q_local = (double *)malloc((local_rows_sz > 0 ? local_rows_sz : 1) * sizeof(double));
		col_p_full = (double *)malloc((n > 0 ? n : 1) * sizeof(double));
		col_q_full = (double *)malloc((n > 0 ? n : 1) * sizeof(double));
		diag_local = (double *)malloc((local_rows_sz > 0 ? local_rows_sz : 1) * sizeof(double));
		if (!A_local || !V_local || !col_p_local || !col_q_local || !col_p_full || !col_q_full || !diag_local) {
			local_ok = 0;
		}
	}

	MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, comm);
	if (!global_ok) {
		free(row_counts);
		free(row_displs);
		free(elem_counts);
		free(elem_displs);
		free(packed_global);
		free(A_local);
		free(V_local);
		free(col_p_local);
		free(col_q_local);
		free(col_p_full);
		free(col_q_full);
		free(diag_local);
		return -1;
	}

	MPI_Scatterv(
		packed_global,
		elem_counts,
		elem_displs,
		MPI_DOUBLE,
		A_local,
		(int)(local_rows_sz * n),
		MPI_DOUBLE,
		root,
		comm
	);

	for (size_t local_row = 0; local_row < local_rows_sz; ++local_row) {
		size_t global_row = (size_t)local_row_start + local_row;
		for (size_t col = 0; col < n; ++col) {
			V_local[local_row * n + col] = (global_row == col) ? 1.0 : 0.0;
		}
	}

	for (int sweep = 0; sweep < maxiter; ++sweep) {
		double local_max_off = 0.0;
		double max_off = 0.0;

		for (size_t local_row = 0; local_row < local_rows_sz; ++local_row) {
			size_t global_row = (size_t)local_row_start + local_row;
			for (size_t col = global_row + 1; col < n; ++col) {
				double val = fabs(A_local[local_row * n + col]);
				if (val > local_max_off) local_max_off = val;
			}
		}

		MPI_Allreduce(&local_max_off, &max_off, 1, MPI_DOUBLE, MPI_MAX, comm);
		if (rank == root && g_jacobi_parallel_debug && (sweep == 0 || sweep % 10 == 0 || max_off <= tol)) {
			fprintf(stderr, "[jacobi_par_mpi] sweep=%d max_off=%.6e tol=%.6e\n", sweep, max_off, tol);
		}
		if (max_off <= tol) {
			converged = 1;
			break;
		}

		for (size_t p = 0; p < n - 1; ++p) {
			int owner_p = owner_of_row(p, n, size);
			size_t local_p = p - (size_t)row_displs[owner_p];

			for (size_t q = p + 1; q < n; ++q) {
				int owner_q = owner_of_row(q, n, size);
				size_t local_q = q - (size_t)row_displs[owner_q];
				double app = 0.0;
				double aqq = 0.0;
				double apq = 0.0;

				if (rank == owner_p) {
					app = A_local[local_p * n + p];
					apq = A_local[local_p * n + q];
				}
				if (rank == owner_q) {
					aqq = A_local[local_q * n + q];
				}

				MPI_Bcast(&app, 1, MPI_DOUBLE, owner_p, comm);
				MPI_Bcast(&apq, 1, MPI_DOUBLE, owner_p, comm);
				MPI_Bcast(&aqq, 1, MPI_DOUBLE, owner_q, comm);

				if (fabs(apq) <= tol) continue;

				double tau = (aqq - app) / (2.0 * apq);
				double t = (tau >= 0.0)
					? 1.0 / (tau + sqrt(1.0 + tau * tau))
					: -1.0 / (-tau + sqrt(1.0 + tau * tau));
				double c = 1.0 / sqrt(1.0 + t * t);
				double s = t * c;
				double app_new = app - t * apq;
				double aqq_new = aqq + t * apq;

				for (size_t local_row = 0; local_row < local_rows_sz; ++local_row) {
					size_t global_row = (size_t)local_row_start + local_row;
					if (global_row == p || global_row == q) continue;

					double aip = A_local[local_row * n + p];
					double aiq = A_local[local_row * n + q];
					A_local[local_row * n + p] = c * aip - s * aiq;
					A_local[local_row * n + q] = s * aip + c * aiq;
				}

				if (rank == owner_p) {
					A_local[local_p * n + p] = app_new;
					A_local[local_p * n + q] = 0.0;
				}
				if (rank == owner_q) {
					A_local[local_q * n + p] = 0.0;
					A_local[local_q * n + q] = aqq_new;
				}

				for (size_t local_row = 0; local_row < local_rows_sz; ++local_row) {
					col_p_local[local_row] = A_local[local_row * n + p];
					col_q_local[local_row] = A_local[local_row * n + q];
				}

				MPI_Allgatherv(col_p_local, local_rows, MPI_DOUBLE, col_p_full, row_counts, row_displs, MPI_DOUBLE, comm);
				MPI_Allgatherv(col_q_local, local_rows, MPI_DOUBLE, col_q_full, row_counts, row_displs, MPI_DOUBLE, comm);

				if (rank == owner_p) {
					memcpy(A_local + local_p * n, col_p_full, n * sizeof(double));
				}
				if (rank == owner_q) {
					memcpy(A_local + local_q * n, col_q_full, n * sizeof(double));
				}

				for (size_t local_row = 0; local_row < local_rows_sz; ++local_row) {
					double vip = V_local[local_row * n + p];
					double viq = V_local[local_row * n + q];
					V_local[local_row * n + p] = c * vip - s * viq;
					V_local[local_row * n + q] = s * vip + c * viq;
				}
			}
		}
	}

	for (size_t local_row = 0; local_row < local_rows_sz; ++local_row) {
		size_t global_row = (size_t)local_row_start + local_row;
		diag_local[local_row] = A_local[local_row * n + global_row];
	}

	if (rank == root) {
		diag_global = (double *)malloc(n * sizeof(double));
		eigvecs_rowmajor = (double *)malloc(n * n * sizeof(double));
		if (!diag_global || !eigvecs_rowmajor) {
			free(diag_global);
			free(eigvecs_rowmajor);
			free(row_counts);
			free(row_displs);
			free(elem_counts);
			free(elem_displs);
			free(packed_global);
			free(A_local);
			free(V_local);
			free(col_p_local);
			free(col_q_local);
			free(col_p_full);
			free(col_q_full);
			free(diag_local);
			return -1;
		}
	}

	MPI_Gatherv(diag_local, local_rows, MPI_DOUBLE, diag_global, row_counts, row_displs, MPI_DOUBLE, root, comm);
	MPI_Gatherv(V_local, (int)(local_rows_sz * n), MPI_DOUBLE, eigvecs_rowmajor, elem_counts, elem_displs, MPI_DOUBLE, root, comm);

	int rc = converged ? 0 : 1;
	if (rank == root) {
		for (size_t i = 0; i < n; ++i) eigvals[i] = diag_global[i];
		unpack_rowmajor_to_colmajor(eigvecs_rowmajor, n, eigvecs);
		jacobi_sort_eigenpairs(eigvals, eigvecs, n);
		if (!converged && g_jacobi_parallel_debug) {
			fprintf(stderr, "[jacobi_par_mpi] reached maxiter=%d before tol=%.6e\n", maxiter, tol);
		}
	}

	MPI_Bcast(&rc, 1, MPI_INT, root, comm);
	if (rc >= 0) {
		MPI_Bcast(eigvals, (int)n, MPI_DOUBLE, root, comm);
		MPI_Bcast(eigvecs, (int)(n * n), MPI_DOUBLE, root, comm);
	}

	free(row_counts);
	free(row_displs);
	free(elem_counts);
	free(elem_displs);
	free(packed_global);
	free(A_local);
	free(V_local);
	free(col_p_local);
	free(col_q_local);
	free(col_p_full);
	free(col_q_full);
	free(diag_local);
	free(diag_global);
	free(eigvecs_rowmajor);
	return rc;
}

int jacobi_eigen_parallel_gather(
	const double *C_local,
	size_t d,
	double *eigvals,
	double *eigvecs,
	MPI_Comm comm,
	int root,
	double tol,
	int maxiter
) {
	if (!C_local || d == 0 || !eigvals || !eigvecs) return -1;

	int rank = 0;
	MPI_Comm_rank(comm, &rank);

	double *C_global = NULL;
	if (rank == root) {
		C_global = (double *)malloc(d * d * sizeof(double));
		if (!C_global) {
			MPI_Abort(comm, MPI_ERR_NO_MEM);
			return -1;
		}
	}

	MPI_Reduce((void *)C_local, C_global, (int)(d * d), MPI_DOUBLE, MPI_SUM, root, comm);
	int rc = jacobi_eigen_parallel_distributed_root(C_global, d, eigvals, eigvecs, comm, root, tol, maxiter);

	if (rank == root) free(C_global);
	return rc;
}

int jacobi_eigen_parallel_gather_root_serial(const double *C_local, size_t d, double *eigvals, double *eigvecs, MPI_Comm comm, int root, double tol, int maxiter) {
	if (!C_local || d == 0 || !eigvals || !eigvecs) return -1;
	int rank = 0;
	int rc = 0;
	double *C_global = NULL;

	MPI_Comm_rank(comm, &rank);
	if (rank == root) {
		C_global = (double *)malloc(d * d * sizeof(double));
	}

	MPI_Barrier(comm);
	if (rank == root && !C_global) {
		MPI_Abort(comm, MPI_ERR_NO_MEM);
		return -1;
	}

	MPI_Reduce((void *)C_local, C_global, (int)(d * d), MPI_DOUBLE, MPI_SUM, root, comm);

	if (rank == root) {
		rc = jacobi_eigen_serial(C_global, d, eigvals, eigvecs, tol, maxiter);
		if (rc != 0) {
			free(C_global);
			C_global = NULL;
		}
	}

	MPI_Bcast(&rc, 1, MPI_INT, root, comm);
	if (rc != 0) return rc;

	MPI_Bcast(eigvals, (int)d, MPI_DOUBLE, root, comm);
	MPI_Bcast(eigvecs, (int)(d * d), MPI_DOUBLE, root, comm);

	if (rank == root) free(C_global);
	return 0;
}

int jacobi_eigen_parallel_gather_root_omp(const double *C_local, size_t d, double *eigvals, double *eigvecs, MPI_Comm comm, int root, double tol, int maxiter) {
	if (!C_local || d == 0 || !eigvals || !eigvecs) return -1;
	int rank = 0;
	int rc = 0;
	double *C_global = NULL;

	MPI_Comm_rank(comm, &rank);
	if (rank == root) {
		C_global = (double *)malloc(d * d * sizeof(double));
	}

	MPI_Barrier(comm);
	if (rank == root && !C_global) {
		MPI_Abort(comm, MPI_ERR_NO_MEM);
		return -1;
	}

	MPI_Reduce((void *)C_local, C_global, (int)(d * d), MPI_DOUBLE, MPI_SUM, root, comm);

	if (rank == root) {
		rc = jacobi_eigen_root_omp(C_global, d, eigvals, eigvecs, tol, maxiter);
		if (rc != 0) {
			free(C_global);
			C_global = NULL;
		}
	}

	MPI_Bcast(&rc, 1, MPI_INT, root, comm);
	if (rc != 0) return rc;

	MPI_Bcast(eigvals, (int)d, MPI_DOUBLE, root, comm);
	MPI_Bcast(eigvecs, (int)(d * d), MPI_DOUBLE, root, comm);

	if (rank == root) free(C_global);
	return 0;
}
