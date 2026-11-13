// 01 / autograd: the operations a Transformer needs, each with its backward pass.
// All tensors are 2-D: [rows, cols]. One sequence is processed at a time:
// rows = positions, cols = features.
import { DType, Tensor, make } from './tensor.js';

function sameShape(a, b, op) {
  if (a.shape[0] !== b.shape[0] || a.shape[1] !== b.shape[1]) {
    throw new Error(`${op}: shape mismatch [${a.shape}] vs [${b.shape}]`);
  }
}

/** a[n,k] @ b[k,m] -> [n,m] */
export function matmul(a, b) {
  const [n, k] = a.shape;
  const [k2, m] = b.shape;
  if (k !== k2) throw new Error(`matmul: inner dims differ ${k} vs ${k2}`);
  const out = new DType(n * m);
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const av = a.data[i * k + p];
      const bo = p * m;
      const oo = i * m;
      for (let j = 0; j < m; j++) out[oo + j] += av * b.data[bo + j];
    }
  }
  return make(out, [n, m], [a, b], (g) => {
    for (let i = 0; i < n; i++) {
      for (let p = 0; p < k; p++) {
        let s = 0;
        const av = a.data[i * k + p];
        for (let j = 0; j < m; j++) {
          s += g[i * m + j] * b.data[p * m + j];
          b.grad[p * m + j] += av * g[i * m + j];
        }
        a.grad[i * k + p] += s;
      }
    }
  });
}

/** a[n,k] @ b[m,k]^T -> [n,m]   (used for Q.K^T and for the tied output layer) */
export function matmulBT(a, b) {
  const [n, k] = a.shape;
  const [m, k2] = b.shape;
  if (k !== k2) throw new Error(`matmulBT: inner dims differ ${k} vs ${k2}`);
  const out = new DType(n * m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += a.data[i * k + p] * b.data[j * k + p];
      out[i * m + j] = s;
    }
  }
  return make(out, [n, m], [a, b], (g) => {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        const gv = g[i * m + j];
        if (gv === 0) continue;
        for (let p = 0; p < k; p++) {
          a.grad[i * k + p] += gv * b.data[j * k + p];
          b.grad[j * k + p] += gv * a.data[i * k + p];
        }
      }
    }
  });
}

/** elementwise a + b (same shape). Used for residual connections and positional encoding. */
export function add(a, b) {
  sameShape(a, b, 'add');
  const out = new DType(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] + b.data[i];
  return make(out, [...a.shape], [a, b], (g) => {
    for (let i = 0; i < g.length; i++) {
      a.grad[i] += g[i];
      b.grad[i] += g[i];
    }
  });
}

/** a[n,m] + bias[1,m], bias is broadcast over rows */
export function addBias(a, bias) {
  const [n, m] = a.shape;
  if (bias.shape[0] !== 1 || bias.shape[1] !== m) throw new Error('addBias: bias must be [1,cols]');
  const out = new DType(n * m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[i * m + j] = a.data[i * m + j] + bias.data[j];
  return make(out, [n, m], [a, bias], (g) => {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        a.grad[i * m + j] += g[i * m + j];
        bias.grad[j] += g[i * m + j];
      }
    }
  });
}

/** elementwise a * b (same shape) */
export function mul(a, b) {
  sameShape(a, b, 'mul');
  const out = new DType(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] * b.data[i];
  return make(out, [...a.shape], [a, b], (g) => {
    for (let i = 0; i < g.length; i++) {
      a.grad[i] += g[i] * b.data[i];
      b.grad[i] += g[i] * a.data[i];
    }
  });
}

/** a * constant */
export function scale(a, s) {
  const out = new DType(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] * s;
  return make(out, [...a.shape], [a], (g) => {
    for (let i = 0; i < g.length; i++) a.grad[i] += g[i] * s;
  });
}

/** sum of all elements -> [1,1] */
export function sum(a) {
  let s = 0;
  for (let i = 0; i < a.data.length; i++) s += a.data[i];
  return make(new DType([s]), [1, 1], [a], (g) => {
    for (let i = 0; i < a.grad.length; i++) a.grad[i] += g[0];
  });
}

export function relu(a) {
  const out = new DType(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] > 0 ? a.data[i] : 0;
  return make(out, [...a.shape], [a], (g) => {
    for (let i = 0; i < g.length; i++) if (a.data[i] > 0) a.grad[i] += g[i];
  });
}

/**
 * Row-wise softmax. `mask` (optional) is a Uint8Array [rows*cols]; 1 = keep, 0 = block.
 * Blocked positions get probability exactly 0. A fully blocked row becomes all zeros.
 */
export function softmax(a, mask = null) {
  const [n, m] = a.shape;
  const out = new DType(n * m);
  for (let i = 0; i < n; i++) {
    const o = i * m;
    let max = -Infinity;
    for (let j = 0; j < m; j++) if (!mask || mask[o + j]) max = Math.max(max, a.data[o + j]);
    if (max === -Infinity) continue;
    let s = 0;
    for (let j = 0; j < m; j++) {
      if (!mask || mask[o + j]) {
        const e = Math.exp(a.data[o + j] - max);
        out[o + j] = e;
        s += e;
      }
    }
    for (let j = 0; j < m; j++) out[o + j] /= s;
  }
  return make(out, [n, m], [a], (g) => {
    for (let i = 0; i < n; i++) {
      const o = i * m;
      let dot = 0;
      for (let j = 0; j < m; j++) dot += g[o + j] * out[o + j];
      for (let j = 0; j < m; j++) a.grad[o + j] += out[o + j] * (g[o + j] - dot);
    }
  });
}

/** Layer norm over each row, then * gamma + beta (gamma, beta are [1,cols]). */
export function layerNorm(x, gamma, beta, eps = 1e-5) {
  const [n, m] = x.shape;
  const out = new DType(n * m);
  const xhat = new DType(n * m);
  const invStd = new DType(n);
  for (let i = 0; i < n; i++) {
    const o = i * m;
    let mean = 0;
    for (let j = 0; j < m; j++) mean += x.data[o + j];
    mean /= m;
    let v = 0;
    for (let j = 0; j < m; j++) v += (x.data[o + j] - mean) ** 2;
    v /= m;
    const inv = 1 / Math.sqrt(v + eps);
    invStd[i] = inv;
    for (let j = 0; j < m; j++) {
      const h = (x.data[o + j] - mean) * inv;
      xhat[o + j] = h;
      out[o + j] = h * gamma.data[j] + beta.data[j];
    }
  }
  return make(out, [n, m], [x, gamma, beta], (g) => {
    for (let i = 0; i < n; i++) {
      const o = i * m;
      let sumD = 0;
      let sumDX = 0;
      for (let j = 0; j < m; j++) {
        const dxh = g[o + j] * gamma.data[j];
        sumD += dxh;
        sumDX += dxh * xhat[o + j];
        gamma.grad[j] += g[o + j] * xhat[o + j];
        beta.grad[j] += g[o + j];
      }
      for (let j = 0; j < m; j++) {
        const dxh = g[o + j] * gamma.data[j];
        x.grad[o + j] += invStd[i] * (dxh - sumD / m - (xhat[o + j] * sumDX) / m);
      }
    }
  });
}

/** Look up rows of table[V,D] by integer ids -> [ids.length, D] */
export function embedding(table, ids) {
  const D = table.shape[1];
  const n = ids.length;
  const out = new DType(n * D);
  for (let i = 0; i < n; i++) {
    if (ids[i] < 0 || ids[i] >= table.shape[0]) throw new Error(`embedding: id ${ids[i]} out of range`);
    for (let j = 0; j < D; j++) out[i * D + j] = table.data[ids[i] * D + j];
  }
  return make(out, [n, D], [table], (g) => {
    for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) table.grad[ids[i] * D + j] += g[i * D + j];
  });
}

/** columns [start, end) of a -> [rows, end-start]   (used to split heads) */
export function sliceCols(a, start, end) {
  const [n, m] = a.shape;
  const w = end - start;
  const out = new DType(n * w);
  for (let i = 0; i < n; i++) for (let j = 0; j < w; j++) out[i * w + j] = a.data[i * m + start + j];
  return make(out, [n, w], [a], (g) => {
    for (let i = 0; i < n; i++) for (let j = 0; j < w; j++) a.grad[i * m + start + j] += g[i * w + j];
  });
}

/** glue tensors side by side along the columns   (used to join heads) */
export function concatCols(list) {
  const n = list[0].shape[0];
  const total = list.reduce((s, t) => s + t.shape[1], 0);
  const out = new DType(n * total);
  let off = 0;
  for (const t of list) {
    if (t.shape[0] !== n) throw new Error('concatCols: row counts differ');
    const w = t.shape[1];
    for (let i = 0; i < n; i++) for (let j = 0; j < w; j++) out[i * total + off + j] = t.data[i * w + j];
    off += w;
  }
  return make(out, [n, total], list, (g) => {
    let o = 0;
    for (const t of list) {
      const w = t.shape[1];
      for (let i = 0; i < n; i++) for (let j = 0; j < w; j++) t.grad[i * w + j] += g[i * total + o + j];
      o += w;
    }
  });
}

/**
 * Softmax + negative log-likelihood, averaged over rows whose target is not `ignore`.
 * logits[n,V], targets = array of n integers. Returns a [1,1] tensor.
 */
export function crossEntropy(logits, targets, ignore = -100) {
  const [n, V] = logits.shape;
  const probs = new DType(n * V);
  let loss = 0;
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (targets[i] === ignore) continue;
    const o = i * V;
    let max = -Infinity;
    for (let j = 0; j < V; j++) max = Math.max(max, logits.data[o + j]);
    let s = 0;
    for (let j = 0; j < V; j++) {
      probs[o + j] = Math.exp(logits.data[o + j] - max);
      s += probs[o + j];
    }
    for (let j = 0; j < V; j++) probs[o + j] /= s;
    loss += Math.log(s) + max - logits.data[o + targets[i]];
    count++;
  }
  if (count === 0) throw new Error('crossEntropy: every target is ignored');
  return make(new DType([loss / count]), [1, 1], [logits], (g) => {
    for (let i = 0; i < n; i++) {
      if (targets[i] === ignore) continue;
      const o = i * V;
      for (let j = 0; j < V; j++) {
        logits.grad[o + j] += (g[0] * (probs[o + j] - (j === targets[i] ? 1 : 0))) / count;
      }
    }
  });
}

export { Tensor };
