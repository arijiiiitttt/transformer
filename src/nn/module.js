// Tiny base class. A Module owns parameter Tensors and/or child Modules.
// namedParameters() finds them by walking the object's fields, so each block
// only has to assign its weights to `this`.
import { Tensor } from '../autograd/tensor.js';

export class Module {
  namedParameters(prefix = '') {
    const out = [];
    for (const [key, value] of Object.entries(this)) {
      if (value instanceof Tensor) {
        out.push([prefix + key, value]);
      } else if (value instanceof Module) {
        out.push(...value.namedParameters(`${prefix}${key}.`));
      } else if (Array.isArray(value)) {
        value.forEach((v, i) => {
          if (v instanceof Module) out.push(...v.namedParameters(`${prefix}${key}.${i}.`));
        });
      }
    }
    return out;
  }

  /** Unique parameters. A weight shared by several blocks (tied embeddings) appears once. */
  parameters() {
    const seen = new Set();
    const list = [];
    for (const [name, tensor] of this.namedParameters()) {
      if (seen.has(tensor)) continue;
      seen.add(tensor);
      list.push([name, tensor]);
    }
    return list;
  }

  zeroGrad() {
    for (const [, t] of this.parameters()) t.zeroGrad();
  }

  countParameters() {
    return this.parameters().reduce((s, [, t]) => s + t.data.length, 0);
  }
}
