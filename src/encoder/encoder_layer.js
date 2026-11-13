// One block of the left stack (the grey box marked Nx on the left):
//   Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm
import { AddNorm } from '../addnorm/add_norm.js';
import { MultiHeadAttention } from '../attention/multi_head.js';
import { FeedForward } from '../feedforward/feed_forward.js';
import { Module } from '../nn/module.js';

export class EncoderLayer extends Module {
  constructor(cfg, rng) {
    super();
    this.selfAttn = new MultiHeadAttention(cfg.dModel, cfg.heads, rng);
    this.norm1 = new AddNorm(cfg.dModel);
    this.ffn = new FeedForward(cfg.dModel, cfg.dFF, rng);
    this.norm2 = new AddNorm(cfg.dModel);
  }

  forward(x, srcMask) {
    x = this.norm1.forward(x, this.selfAttn.forward(x, x, srcMask));
    return this.norm2.forward(x, this.ffn.forward(x));
  }
}
