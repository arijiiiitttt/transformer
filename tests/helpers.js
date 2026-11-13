import { Tensor } from '../src/autograd/tensor.js';
import { mul, sum } from '../src/autograd/ops.js';
import { mulberry32, randn } from '../src/utils/rng.js';

export function randomTensor(rows, cols, seed = 1, std = 1) {
  const rng = mulberry32(seed);
  const t = Tensor.zeros(rows, cols);
  for (let i = 0; i < t.data.length; i++) t.data[i] = randn(rng) * std;
  return t;
}

/**
 * Numerical gradient check. build() must return a [1,1] loss tensor using `inputs`.
 * Compares analytic gradients with central finite differences on up to `maxChecks`
 * entries per input. Returns the worst relative error.
 */
export function gradCheck(build, inputs, { eps = 1e-5, maxChecks = 12 } = {}) {
  for (const t of inputs) t.zeroGrad();
  build().backward();
  const analytic = inputs.map((t) => Float64Array.from(t.grad));
  let worst = 0;
  inputs.forEach((t, k) => {
    const step = Math.max(1, Math.floor(t.data.length / maxChecks));
    for (let i = 0; i < t.data.length; i += step) {
      const orig = t.data[i];
      t.data[i] = orig + eps;
      const up = build().item();
      t.data[i] = orig - eps;
      const down = build().item();
      t.data[i] = orig;
      const numeric = (up - down) / (2 * eps);
      const err = Math.abs(numeric - analytic[k][i]) / Math.max(1e-6, Math.abs(numeric) + Math.abs(analytic[k][i]));
      worst = Math.max(worst, err);
    }
  });
  return worst;
}

/** turn any tensor into a scalar loss with fixed random weights so every output element matters */
export function weightedSum(out, seed = 99) {
  return sum(mul(out, randomTensor(out.rows, out.cols, seed)));
}
