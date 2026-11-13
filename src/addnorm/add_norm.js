// Diagram block: "Add & Norm".
// Add the sub-layer's input back to its output (the residual arrow that skips
// around each box), then layer-normalise. Post-norm, exactly as in the paper:
//   LayerNorm(x + Sublayer(x))
import { add, layerNorm } from '../autograd/ops.js';
import { Module } from '../nn/module.js';
import { ones, zeros } from '../utils/init.js';

export class AddNorm extends Module {
  constructor(dModel) {
    super();
    this.gamma = ones(1, dModel);
    this.beta = zeros(1, dModel);
  }

  /** x: input of the sub-layer, sublayerOut: its output */
  forward(x, sublayerOut) {
    return layerNorm(add(x, sublayerOut), this.gamma, this.beta);
  }
}
