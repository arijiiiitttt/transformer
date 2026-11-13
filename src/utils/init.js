// Weight initialisers. Each returns a Tensor that is a trainable parameter.
import { DType, Tensor } from '../autograd/tensor.js';
import { randn } from './rng.js';

/** Glorot / Xavier uniform: keeps signal size steady through a linear layer. */
export function xavier(rows, cols, rng) {
  const limit = Math.sqrt(6 / (rows + cols));
  const t = Tensor.zeros(rows, cols);
  for (let i = 0; i < t.data.length; i++) t.data[i] = (rng() * 2 - 1) * limit;
  return t;
}

export function normal(rows, cols, std, rng) {
  const t = Tensor.zeros(rows, cols);
  for (let i = 0; i < t.data.length; i++) t.data[i] = randn(rng) * std;
  return t;
}

export function zeros(rows, cols) {
  return Tensor.zeros(rows, cols);
}

export function ones(rows, cols) {
  const t = Tensor.zeros(rows, cols);
  t.data.fill(1);
  return t;
}

export { DType };
