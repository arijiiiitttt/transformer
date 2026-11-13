// The core of attention:  softmax(Q K^T / sqrt(d_k)) V
// Each query row scores every key, the scores become weights that sum to 1,
// and the output is that weighted mix of the value rows.
import { matmul, matmulBT, scale, softmax } from '../autograd/ops.js';

/**
 * Q [nq, dk], K [nk, dk], V [nk, dv], mask Uint8Array [nq*nk] or null.
 * Returns { out [nq, dv], weights [nq, nk] }.
 */
export function scaledDotProductAttention(Q, K, V, mask = null) {
  const dk = Q.cols;
  const scores = scale(matmulBT(Q, K), 1 / Math.sqrt(dk));
  const weights = softmax(scores, mask);
  return { out: matmul(weights, V), weights };
}
