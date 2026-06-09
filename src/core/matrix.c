#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <limits.h>
#include <sys/types.h>

#include <mpi.h>

#include <cblas.h>

static int to_mpi_count(size_t value) {
	return value <= (size_t)INT_MAX ? (int)value : -1;
}

void matrix_row_block(size_t rows, int nranks, int rank, size_t *start, size_t *count) {
	size_t base = rows / (size_t)nranks;
	size_t rem = rows % (size_t)nranks;
	size_t local = base + ((size_t)rank < rem ? 1u : 0u);
	size_t offset = (size_t)rank * base + ((size_t)rank < rem ? (size_t)rank : rem);
	if (start) *start = offset;
	if (count) *count = local;
}

matrix_t *matrix_create(size_t rows, size_t cols) {
	matrix_t *m = (matrix_t *)malloc(sizeof(matrix_t));
	if (!m) return NULL;
	m->rows = rows;
	m->cols = cols;
	size_t n = rows * cols;
	m->data = NULL;
	if (n > 0) {
		m->data = (double *)malloc(n * sizeof(double));
		if (!m->data) {
			free(m);
			return NULL;
		}
		memset(m->data, 0, n * sizeof(double));
	}
	return m;
}

void matrix_free(matrix_t *m) {
	if (!m) return;
	free(m->data);
	free(m);
}

double *matrix_data(matrix_t *m) {
	if (!m) return NULL;
	return m->data;
}

int matrix_clear(matrix_t *m) {
	if (!m || !m->data) return -1;
	memset(m->data, 0, m->rows * m->cols * sizeof(double));
	return 0;
}

int matrix_copy(matrix_t *dst, const matrix_t *src) {
	if (!dst || !src) return -1;
	if (dst->rows != src->rows || dst->cols != src->cols) return -1;
	size_t n = src->rows * src->cols;
	memcpy(dst->data, src->data, n * sizeof(double));
	return 0;
}

int matrix_dgemm(double alpha, const matrix_t *A, const matrix_t *B, double beta, matrix_t *C) {
	if (!A || !B || !C) return -1;
	if (A->cols != B->rows) return -1;
	if (C->rows != A->rows || C->cols != B->cols) return -1;
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
				(int)A->rows, (int)B->cols, (int)A->cols,
				alpha, A->data, (int)A->rows,
				B->data, (int)B->rows,
				beta, C->data, (int)C->rows);
	return 0;
}

int matrix_local_covariance(const matrix_t *X_local, matrix_t *C_local) {
	if (!X_local || !C_local) return -1;
	if (C_local->rows != C_local->cols) return -1;
	if (C_local->rows != X_local->cols) return -1;
	size_t n = X_local->rows;
	size_t d = X_local->cols;
	/* C = X^T * X (d x d) */
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
				(int)d, (int)d, (int)n,
				1.0, X_local->data, (int)X_local->rows,
				X_local->data, (int)X_local->rows,
				0.0, C_local->data, (int)C_local->rows);
	return 0;
}

void matrix_center_columns(matrix_t *X, const double *mu) {
	if (!X || !mu) return;
	size_t n = X->rows;
	size_t d = X->cols;
	for (size_t j = 0; j < d; ++j) {
		double m = mu[j];
		double *col = X->data + j * n;
		for (size_t i = 0; i < n; ++i) col[i] -= m;
	}
}

int matrix_read_csv(const char *path, matrix_t **out) {
	if (!path || !out) return -1;
	FILE *f = fopen(path, "r");
	if (!f) return -1;
	char *line = NULL;
	size_t len = 0;
	ssize_t read;
	size_t rows = 0, cols = 0;
	double *vals = NULL;
	size_t cap = 0, total = 0;
	while ((read = getline(&line, &len, f)) != -1) {
		char *p = line;
		while (*p && isspace((unsigned char)*p)) ++p;
		if (*p == '\0' || *p == '#') continue;
		char *saveptr = NULL;
		char *tok = strtok_r(line, ", \t\r\n", &saveptr);
		size_t cnt = 0;
		while (tok) {
			char *endptr = NULL;
			errno = 0;
			double v = strtod(tok, &endptr);
			if (endptr == tok) {
				free(vals);
				free(line);
				fclose(f);
				return -1;
			}
			if (total >= cap) {
				cap = cap ? cap * 2 : 1024;
				double *tmp = (double *)realloc(vals, cap * sizeof(double));
				if (!tmp) {
					free(vals);
					free(line);
					fclose(f);
					return -1;
				}
				vals = tmp;
			}
			vals[total++] = v;
			++cnt;
			tok = strtok_r(NULL, ", \t\r\n", &saveptr);
		}
		if (cnt == 0) continue;
		if (rows == 0) cols = cnt;
		else if (cnt != cols) {
			free(vals);
			free(line);
			fclose(f);
			return -1;
		}
		++rows;
	}
	free(line);
	fclose(f);
	if (rows == 0 || cols == 0) {
		free(vals);
		return -1;
	}
	matrix_t *m = matrix_create(rows, cols);
	if (!m) {
		free(vals);
		return -1;
	}
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			m->data[j * rows + i] = vals[i * cols + j];
		}
	}
	free(vals);
	*out = m;
	return 0;
}

int matrix_write_csv(const char *path, const matrix_t *m) {
	if (!path || !m) return -1;
	FILE *f = fopen(path, "w");
	if (!f) return -1;
	size_t n = m->rows;
	size_t d = m->cols;
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < d; ++j) {
			double v = m->data[j * n + i];
			if (j + 1 < d) fprintf(f, "%.12e,", v);
			else fprintf(f, "%.12e", v);
		}
		fprintf(f, "\n");
	}
	fclose(f);
	return 0;
}

int matrix_scatter_rows_from_root(const matrix_t *global, matrix_t **local_out, int root, MPI_Comm comm) {
	if (!local_out) return -1;

	int rank = 0;
	int size = 1;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	unsigned long long dims[2] = {0ULL, 0ULL};
	if (rank == root) {
		if (!global) return -1;
		dims[0] = (unsigned long long)global->rows;
		dims[1] = (unsigned long long)global->cols;
	}
	MPI_Bcast(dims, 2, MPI_UNSIGNED_LONG_LONG, root, comm);

	size_t rows = (size_t)dims[0];
	size_t cols = (size_t)dims[1];
	size_t start = 0;
	size_t local_rows = 0;
	matrix_row_block(rows, size, rank, &start, &local_rows);

	matrix_t *local = matrix_create(local_rows, cols);
	if (!local) return -1;

	if (rank == root) {
		for (int target = 0; target < size; ++target) {
			size_t target_start = 0;
			size_t target_rows = 0;
			matrix_row_block(rows, size, target, &target_start, &target_rows);
			if (target == root) {
				for (size_t j = 0; j < cols; ++j) {
					memcpy(local->data + j * local_rows,
						   global->data + j * rows + target_start,
						   target_rows * sizeof(double));
				}
				continue;
			}

			size_t packed_elems = target_rows * cols;
			int mpi_count = to_mpi_count(packed_elems);
			if (mpi_count < 0) {
				matrix_free(local);
				return -1;
			}
			if (packed_elems == 0) {
				MPI_Send(NULL, 0, MPI_DOUBLE, target, 100, comm);
				continue;
			}

			double *buffer = (double *)malloc(packed_elems * sizeof(double));
			if (!buffer) {
				matrix_free(local);
				return -1;
			}
			for (size_t j = 0; j < cols; ++j) {
				memcpy(buffer + j * target_rows,
					   global->data + j * rows + target_start,
					   target_rows * sizeof(double));
			}
			MPI_Send(buffer, mpi_count, MPI_DOUBLE, target, 100, comm);
			free(buffer);
		}
	} else {
		size_t packed_elems = local_rows * cols;
		int mpi_count = to_mpi_count(packed_elems);
		if (mpi_count < 0) {
			matrix_free(local);
			return -1;
		}
		if (packed_elems > 0) {
			MPI_Recv(local->data, mpi_count, MPI_DOUBLE, root, 100, comm, MPI_STATUS_IGNORE);
		}
	}

	*local_out = local;
	return 0;
}

int matrix_gather_rows_to_root(const matrix_t *local, matrix_t **global_out, int root, MPI_Comm comm) {
	if (!local) return -1;

	int rank = 0;
	int size = 1;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	unsigned long long local_dims[2] = {
		(unsigned long long)local->rows,
		(unsigned long long)local->cols
	};
	unsigned long long global_rows_ull = 0ULL;
	MPI_Allreduce(&local_dims[0], &global_rows_ull, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);

	size_t rows = (size_t)global_rows_ull;
	size_t cols = (size_t)local_dims[1];

	matrix_t *global = NULL;
	if (rank == root) {
		if (!global_out) return -1;
		global = matrix_create(rows, cols);
		if (!global) return -1;
	}

	size_t root_start = 0;
	size_t root_rows = 0;
	matrix_row_block(rows, size, rank, &root_start, &root_rows);

	if (rank == root) {
		for (size_t j = 0; j < cols; ++j) {
			memcpy(global->data + j * rows + root_start,
				   local->data + j * local->rows,
				   local->rows * sizeof(double));
		}

		for (int source = 0; source < size; ++source) {
			if (source == root) continue;
			size_t source_start = 0;
			size_t source_rows = 0;
			matrix_row_block(rows, size, source, &source_start, &source_rows);

			size_t packed_elems = source_rows * cols;
			int mpi_count = to_mpi_count(packed_elems);
			if (mpi_count < 0) {
				matrix_free(global);
				return -1;
			}
			if (packed_elems == 0) {
				continue;
			}
			double *buffer = (double *)malloc(packed_elems * sizeof(double));
			if (!buffer) {
				matrix_free(global);
				return -1;
			}
			MPI_Recv(buffer, mpi_count, MPI_DOUBLE, source, 101, comm, MPI_STATUS_IGNORE);
			for (size_t j = 0; j < cols; ++j) {
				memcpy(global->data + j * rows + source_start,
					   buffer + j * source_rows,
					   source_rows * sizeof(double));
			}
			free(buffer);
		}

		*global_out = global;
	} else {
		size_t packed_elems = local->rows * cols;
		int mpi_count = to_mpi_count(packed_elems);
		if (mpi_count < 0) return -1;
		if (packed_elems > 0) {
			MPI_Send(local->data, mpi_count, MPI_DOUBLE, root, 101, comm);
		}
	}

	return 0;
}
