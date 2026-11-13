// Diagram block: "Feed Forward".
// Two linear layers with a ReLU between them, applied to every position on its
// own:  FFN(x) = max(0, x W1 + b1) W2 + b2
import { addBias, matmul, relu } from '../autograd/ops.js';
import { Module } from '../nn/module.js';
import { xavier, zeros } from '../utils/init.js';

export class FeedForward extends Module {
  constructor(dModel, dFF, rng) {
    super();
    this.W1 = xavier(dModel, dFF, rng);
    this.b1 = zeros(1, dFF);
    this.W2 = xavier(dFF, dModel, rng);
    this.b2 = zeros(1, dModel);
  }

  forward(x) {
    return addBias(matmul(relu(addBias(matmul(x, this.W1), this.b1)), this.W2), this.b2);
  }
}
