// Adam optimiser, learning-rate schedule and gradient clipping.
import { DType } from '../autograd/tensor.js';

export class Adam {
  constructor(params, { beta1 = 0.9, beta2 = 0.98, eps = 1e-9, weightDecay = 0 } = {}) {
    this.params = params.map(([, t]) => t);
    this.b1 = beta1;
    this.b2 = beta2;
    this.eps = eps;
    this.wd = weightDecay;
    this.t = 0;
    this.m = this.params.map((p) => new DType(p.data.length));
    this.v = this.params.map((p) => new DType(p.data.length));
  }

  /** gradScale divides the accumulated gradients (use 1/batchSize) */
  step(lr, gradScale = 1) {
    this.t++;
    const c1 = 1 - Math.pow(this.b1, this.t);
    const c2 = 1 - Math.pow(this.b2, this.t);
    this.params.forEach((p, k) => {
      const m = this.m[k];
      const v = this.v[k];
      for (let i = 0; i < p.data.length; i++) {
        const g = p.grad[i] * gradScale;
        m[i] = this.b1 * m[i] + (1 - this.b1) * g;
        v[i] = this.b2 * v[i] + (1 - this.b2) * g * g;
        const update = m[i] / c1 / (Math.sqrt(v[i] / c2) + this.eps);
        p.data[i] -= lr * (update + this.wd * p.data[i]);
      }
    });
  }
}

/** Linear warm-up to peakLr over `warmup` steps, then decay with 1/sqrt(step) (the paper's shape). */
export function lrSchedule(step, peakLr, warmup) {
  if (step < warmup) return (peakLr * step) / warmup;
  return peakLr * Math.sqrt(warmup / step);
}

/** Scale all gradients so their global norm is at most maxNorm. Returns the norm before clipping. */
export function clipGradNorm(params, maxNorm, gradScale = 1) {
  let sq = 0;
  for (const [, p] of params) for (let i = 0; i < p.grad.length; i++) sq += (p.grad[i] * gradScale) ** 2;
  const norm = Math.sqrt(sq);
  if (norm > maxNorm) {
    const k = maxNorm / norm;
    for (const [, p] of params) for (let i = 0; i < p.grad.length; i++) p.grad[i] *= k;
  }
  return norm;
}
