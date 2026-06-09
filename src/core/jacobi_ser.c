#include "jacobi.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline size_t IDX(size_t i, size_t j, size_t n) { return j * n + i; }

int jacobi_eigen_serial(double *A, size_t n, double *eigvals, double *eigvecs, double tol, int maxiter) {
	if (!A || n == 0 || !eigvals) return -1;
	int need_vecs = (eigvecs != NULL);
	if (need_vecs) {
		for (size_t j = 0; j < n; ++j) {
			for (size_t i = 0; i < n; ++i) {
				eigvecs[IDX(i, j, n)] = (i == j) ? 1.0 : 0.0;
			}
		}
	}

	for (int iter = 0; iter < maxiter; ++iter) {
		double max_off = 0.0;
		for (size_t p = 0; p < n; ++p) {
			for (size_t q = p + 1; q < n; ++q) {
				double apq = A[IDX(p, q, n)];
				if (fabs(apq) > max_off) max_off = fabs(apq);
			}
		}
		if (max_off <= tol) break;

		for (size_t p = 0; p < n - 1; ++p) {
			for (size_t q = p + 1; q < n; ++q) {
				double apq = A[IDX(p, q, n)];
				if (apq == 0.0) continue;
				double app = A[IDX(p, p, n)];
				double aqq = A[IDX(q, q, n)];
				double tau = (aqq - app) / (2.0 * apq);
				double t;
				if (tau >= 0.0) t = 1.0 / (tau + sqrt(1.0 + tau * tau));
				else t = -1.0 / (-tau + sqrt(1.0 + tau * tau));
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

				if (need_vecs) {
					for (size_t i = 0; i < n; ++i) {
						double vip = eigvecs[IDX(i, p, n)];
						double viq = eigvecs[IDX(i, q, n)];
						eigvecs[IDX(i, p, n)] = c * vip - s * viq;
						eigvecs[IDX(i, q, n)] = s * vip + c * viq;
					}
				}
			}
		}
	}

	/* collect eigenvalues from diagonal */
	double *vals = (double *)malloc(n * sizeof(double));
	if (!vals) return -1;
	for (size_t i = 0; i < n; ++i) vals[i] = A[IDX(i, i, n)];

	/* selection sort descending and reorder eigenvectors if present */
	for (size_t i = 0; i < n; ++i) {
		size_t best = i;
		for (size_t j = i + 1; j < n; ++j) if (vals[j] > vals[best]) best = j;
		if (best != i) {
			double tmp = vals[i]; vals[i] = vals[best]; vals[best] = tmp;
			if (need_vecs) {
				for (size_t r = 0; r < n; ++r) {
					double a = eigvecs[IDX(r, i, n)];
					eigvecs[IDX(r, i, n)] = eigvecs[IDX(r, best, n)];
					eigvecs[IDX(r, best, n)] = a;
				}
			}
		}
	}

	for (size_t i = 0; i < n; ++i) eigvals[i] = vals[i];
	free(vals);
	return 0;
}

