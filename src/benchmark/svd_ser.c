/* svd_ser.c
 * Serial PCA using SVD (LAPACK)
 */

#include <stdio.h>
#include <stdlib.h>
#include "pca.h"
#include "matrix.h"
#include "utils.h"

int main(int argc, char **argv) {
	if (argc < 3) {
		fprintf(stderr, "Usage: %s input.csv k [out_projection.csv]\n", argv[0]);
		return 1;
	}
	const char *input = argv[1];
	int k = atoi(argv[2]);
	const char *outp = (argc >= 4) ? argv[3] : "results/svd_ser_projection.csv";

	double t0 = pca_wtime();
	matrix_t *X = NULL;
	if (matrix_read_csv(input, &X) != 0) {
		fprintf(stderr, "Failed to read %s\n", input);
		return 2;
	}
	double t_read = pca_wtime() - t0;

	size_t n = X->rows, d = X->cols;
	double *mu = (double *)malloc(d * sizeof(double));
	if (!mu) {
		matrix_free(X);
		return 2;
	}
	if (pca_compute_global_mean(X, mu, MPI_COMM_SELF) != 0) {
		fprintf(stderr, "Failed to compute mean\n");
		free(mu);
		matrix_free(X);
		return 2;
	}
	double t_center = pca_wtime();
	matrix_center_columns(X, mu);
	t_center = pca_wtime() - t_center;

	double *singvals = (double *)malloc(d * sizeof(double));
	matrix_t *V = NULL;
	matrix_t *projection = NULL;
	if (!singvals) {
		free(mu);
		matrix_free(X);
		return 2;
	}
	double t_pca_start = pca_wtime();
	int rc = pca_by_svd_serial(X, k, singvals, &V, &projection);
	double t_pca = pca_wtime() - t_pca_start;

	if (rc != 0) {
		fprintf(stderr, "pca_by_svd_serial failed: %d\n", rc);
		free(mu);
		matrix_free(X);
		free(singvals);
		return 3;
	}

	if (matrix_write_csv(outp, projection) != 0) fprintf(stderr, "Failed to write projection to %s\n", outp);

	double explained = pca_explained_ratio_from_singvals(singvals, d, k);
	double t_total = pca_wtime() - t0;
	printf("svd_ser,%zu,%zu,%d,1,%.6f,%.6f,%.6f,%.6f,%.8f\n", n, d, k, t_read, t_center, t_pca, t_total, explained);

	free(mu);
	matrix_free(X);
	matrix_free(V);
	matrix_free(projection);
	free(singvals);
	return 0;
}
