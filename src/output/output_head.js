// Diagram blocks: "Linear" and "Softmax" -> "Output Probabilities".
// Linear maps each decoder vector to one score (logit) per vocabulary token. It
// re-uses the embedding table (weight tying, as in the paper), so it adds no
// new parameters. Softmax turns the scores into probabilities. During training
// the softmax is folded into crossEntropy() for numerical stability.
import { matmulBT, softmax } from '../autograd/ops.js';
import { Module } from '../nn/module.js';

export class OutputHead extends Module {
  constructor(embed) {
    super();
    this.embed = embed;
  }

  /** x [n, d] -> logits [n, vocab] */
  logits(x) {
    return matmulBT(x, this.embed.table);
  }

  /** x [n, d] -> probabilities [n, vocab] (each row sums to 1) */
  probabilities(x) {
    return softmax(this.logits(x));
  }
}
