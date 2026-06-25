#include "jacobi.h"

#include <math.h>
#include <stdlib.h>

static inline size_t IDX(size_t i, size_t j, size_t n) { return j * n + i; }

static void jacobi_init_identity(double *eigvecs, size_t n) {
	for (size_t j = 0; j < n; ++j) {
		for (size_t i = 0; i < n; ++i) {
			eigvecs[IDX(i, j, n)] = (i == j) ? 1.0 : 0.0;
		}
	}
}

static void jacobi_rotate_pair(double *A, size_t n, double *eigvecs, size_t p, size_t q, double tol) {
	double apq = A[IDX(p, q, n)];
	if (fabs(apq) <= tol) return;

	double app = A[IDX(p, p, n)];
	double aqq = A[IDX(q, q, n)];
	double tau = (aqq - app) / (2.0 * apq);
	double t = (tau >= 0.0)
		? 1.0 / (tau + sqrt(1.0 + tau * tau))
		: -1.0 / (-tau + sqrt(1.0 + tau * tau));
	double c = 1.0 / sqrt(1.0 + t * t);
	double s = t * c;

	double app_new = app - t * apq;
	double aqq_new = aqq + t * apq;
	A[IDX(p, p, n)] = app_new;
	A[IDX(q, q, n)] = aqq_new;
	A[IDX(p, q, n)] = 0.0;
	A[IDX(q, p, n)] = 0.0;

	for (size_t i = 0; i < n; ++i) {
		if (i == p || i == q) continue;
		double aip = A[IDX(i, p, n)];
		double aiq = A[IDX(i, q, n)];
		double new_ip = c * aip - s * aiq;
		double new_iq = s * aip + c * aiq;
		A[IDX(i, p, n)] = new_ip;
		A[IDX(i, q, n)] = new_iq;
		A[IDX(p, i, n)] = new_ip;
		A[IDX(q, i, n)] = new_iq;
	}

	if (eigvecs) {
		for (size_t i = 0; i < n; ++i) {
			double vip = eigvecs[IDX(i, p, n)];
			double viq = eigvecs[IDX(i, q, n)];
			eigvecs[IDX(i, p, n)] = c * vip - s * viq;
			eigvecs[IDX(i, q, n)] = s * vip + c * viq;
		}
	}
}

static void jacobi_round_robin_init(size_t *order, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		order[i] = i;
	}
}

static size_t jacobi_round_robin_build_phase(
	const size_t *order,
	size_t n,
	size_t matrix_n,
	size_t *pairs_p,
	size_t *pairs_q
) {
	size_t count = 0;
	for (size_t i = 0; i < n / 2; ++i) {
		size_t a = order[i];
		size_t b = order[n - 1 - i];
		if (a >= matrix_n || b >= matrix_n) continue;
		if (a > b) {
			size_t tmp = a;
			a = b;
			b = tmp;
		}
		pairs_p[count] = a;
		pairs_q[count] = b;
		++count;
	}
	return count;
}

static void jacobi_round_robin_rotate(size_t *order, size_t n) {
	if (n <= 2) return;
	size_t last = order[n - 1];
	for (size_t i = n - 1; i > 1; --i) {
		order[i] = order[i - 1];
	}
	order[1] = last;
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
				for (size_t r = 0; r < n; ++r) {
					double v = eigvecs[IDX(r, i, n)];
					eigvecs[IDX(r, i, n)] = eigvecs[IDX(r, best, n)];
					eigvecs[IDX(r, best, n)] = v;
				}
			}
		}
	}
}

int jacobi_eigen_serial(double *A, size_t n, double *eigvals, double *eigvecs, double tol, int maxiter) {
	if (!A || n == 0 || !eigvals) return -1;
	int need_vecs = (eigvecs != NULL);
	if (need_vecs) {
		jacobi_init_identity(eigvecs, n);
	}

	size_t phase_count = (n < 2) ? 0 : ((n % 2 == 0) ? (n - 1) : n);
	size_t order_n = (n % 2 == 0) ? n : (n + 1);
	size_t *order = (size_t *)malloc(order_n * sizeof(size_t));
	size_t *pairs_p = (size_t *)malloc((order_n / 2) * sizeof(size_t));
	size_t *pairs_q = (size_t *)malloc((order_n / 2) * sizeof(size_t));
	if (!order || !pairs_p || !pairs_q) {
		free(order);
		free(pairs_p);
		free(pairs_q);
		return -1;
	}
	jacobi_round_robin_init(order, order_n);

	for (int iter = 0; iter < maxiter; ++iter) {
		double max_off = 0.0;
		for (size_t p = 0; p < n; ++p) {
			for (size_t q = p + 1; q < n; ++q) {
				double apq = fabs(A[IDX(p, q, n)]);
				if (apq > max_off) max_off = apq;
			}
		}
		if (max_off <= tol) break;

		jacobi_round_robin_init(order, order_n);
		for (size_t phase = 0; phase < phase_count; ++phase) {
			size_t pair_count = jacobi_round_robin_build_phase(order, order_n, n, pairs_p, pairs_q);
			for (size_t idx = 0; idx < pair_count; ++idx) {
				jacobi_rotate_pair(A, n, need_vecs ? eigvecs : NULL, pairs_p[idx], pairs_q[idx], tol);
			}
			jacobi_round_robin_rotate(order, order_n);
		}
	}

	for (size_t i = 0; i < n; ++i) {
		eigvals[i] = A[IDX(i, i, n)];
	}
	jacobi_sort_eigenpairs(eigvals, eigvecs, n);

	free(order);
	free(pairs_p);
	free(pairs_q);
	return 0;
}
