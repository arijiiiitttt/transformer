// Diagram block: "Input Embedding" (left) and "Output Embedding" (right).
// Turns token ids into vectors. As in the paper, the same table is shared by the
// input embedding, the output embedding and the final linear layer, and the
// vectors are multiplied by sqrt(d_model).
import { embedding, scale } from '../autograd/ops.js';
import { Module } from '../nn/module.js';
import { normal } from '../utils/init.js';

export class Embedding extends Module {
  constructor(vocabSize, dModel, rng) {
    super();
    this.dModel = dModel;
    // std = d_model^-0.5, so after the sqrt(d_model) scale the vectors have unit size
    this.table = normal(vocabSize, dModel, dModel ** -0.5, rng);
  }

  /** ids: array of ints -> Tensor [ids.length, dModel] */
  forward(ids) {
    return scale(embedding(this.table, ids), Math.sqrt(this.dModel));
  }
}
