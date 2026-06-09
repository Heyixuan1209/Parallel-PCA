#!/usr/bin/env python3
"""Generate synthetic datasets for PCA experiments."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def generate_data(n: int, d: int, r: int, noise: float, seed: int | None = None) -> np.ndarray:
	rng = np.random.default_rng(seed)
	U = rng.normal(size=(n, r))
	V = rng.normal(size=(d, r))
	Q, _ = np.linalg.qr(V)
	V = Q[:, :r]
	s = np.linspace(1.0, 0.5, r)
	X = U * s[np.newaxis, :] @ V.T
	if noise > 0.0:
		X += noise * rng.normal(size=(n, d))
	return X


def save_csv(path: str, X: np.ndarray) -> None:
	np.savetxt(path, X, delimiter=",", fmt="%.8e")


def save_npy(path: str, X: np.ndarray) -> None:
	np.save(path, X)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate low-rank synthetic data for PCA")
	parser.add_argument("--samples", "-n", type=int, default=1000, help="number of samples (rows)")
	parser.add_argument("--features", "-d", type=int, default=100, help="number of features (columns)")
	parser.add_argument("--rank", "-r", type=int, default=0, help="intrinsic rank, 0 -> min(10, d)")
	parser.add_argument("--noise", type=float, default=0.01, help="Gaussian noise standard deviation")
	parser.add_argument("--seed", type=int, default=2026, help="random seed")
	parser.add_argument("--out", "-o", required=True, help="output file path (.csv or .npy)")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	n = args.samples
	d = args.features
	r = args.rank if args.rank > 0 else min(10, d)
	if r <= 0 or r > min(n, d):
		raise ValueError(f"rank must be in [1, min(n, d)], got r={r}, n={n}, d={d}")

	X = generate_data(n, d, r, args.noise, args.seed)

	out_dir = os.path.dirname(args.out)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)

	if args.out.lower().endswith(".npy"):
		save_npy(args.out, X)
	else:
		save_csv(args.out, X)
	print(f"Saved {n}x{d} synthetic matrix to {args.out}")


if __name__ == "__main__":
	try:
		main()
	except BrokenPipeError:
		sys.exit(0)
