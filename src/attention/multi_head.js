// Diagram block: "Multi-Head Attention" (and "Masked Multi-Head Attention").
// Project the input into Q, K, V, split the columns into `heads` groups, run
// scaled dot-product attention in each group, glue the results back together
// and mix them with a final projection Wo.
import { concatCols, matmul, sliceCols } from '../autograd/ops.js';
import { Module } from '../nn/module.js';
import { xavier } from '../utils/init.js';
import { scaledDotProductAttention } from './scaled_dot_product.js';

export class MultiHeadAttention extends Module {
  constructor(dModel, heads, rng) {
    super();
    if (dModel % heads !== 0) throw new Error(`dModel ${dModel} must be divisible by heads ${heads}`);
    this.dModel = dModel;
    this.heads = heads;
    this.dHead = dModel / heads;
    this.Wq = xavier(dModel, dModel, rng);
    this.Wk = xavier(dModel, dModel, rng);
    this.Wv = xavier(dModel, dModel, rng);
    this.Wo = xavier(dModel, dModel, rng);
    this.lastWeights = null; // per-head attention maps from the latest call (for inspection)
  }

  /**
   * xq  [nq, d]  where the queries come from
   * xkv [nk, d]  where keys and values come from
   *   self-attention : xq === xkv
   *   cross-attention: xq = decoder state, xkv = encoder output
   */
  forward(xq, xkv, mask = null) {
    const Q = matmul(xq, this.Wq);
    const K = matmul(xkv, this.Wk);
    const V = matmul(xkv, this.Wv);
    const outs = [];
    const maps = [];
    for (let h = 0; h < this.heads; h++) {
      const a = h * this.dHead;
      const b = a + this.dHead;
      const { out, weights } = scaledDotProductAttention(
        sliceCols(Q, a, b),
        sliceCols(K, a, b),
        sliceCols(V, a, b),
        mask,
      );
      outs.push(out);
      maps.push(weights);
    }
    this.lastWeights = maps;
    return matmul(concatCols(outs), this.Wo);
  }
}
