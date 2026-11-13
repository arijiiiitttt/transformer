// Diagram block: "Positional Encoding" (the sine-wave circle + the ⊕).
// Attention has no sense of order by itself, so we add a fixed pattern of
// sines and cosines that tells each position where it is:
//   PE(pos, 2i)   = sin(pos / 10000^(2i/d))
//   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
import { DType, Tensor } from '../autograd/tensor.js';
import { add } from '../autograd/ops.js';
import { Module } from '../nn/module.js';

export class PositionalEncoding extends Module {
  constructor(dModel, maxLen) {
    super();
    if (dModel % 2 !== 0) throw new Error('dModel must be even for sinusoidal encoding');
    this.dModel = dModel;
    this.maxLen = maxLen;
    // plain typed array (not a Tensor): it is a constant, not a trained weight
    this.table = new DType(maxLen * dModel);
    for (let pos = 0; pos < maxLen; pos++) {
      for (let i = 0; i < dModel / 2; i++) {
        const angle = pos / Math.pow(10000, (2 * i) / dModel);
        this.table[pos * dModel + 2 * i] = Math.sin(angle);
        this.table[pos * dModel + 2 * i + 1] = Math.cos(angle);
      }
    }
  }

  /** x: [n, dModel] -> x + PE[0..n) */
  forward(x) {
    const n = x.rows;
    if (n > this.maxLen) throw new Error(`sequence length ${n} is longer than maxLen ${this.maxLen}`);
    const pe = new Tensor(this.table.slice(0, n * this.dModel), [n, this.dModel]);
    return add(x, pe);
  }
}
