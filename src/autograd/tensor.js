// 01 / autograd: the Tensor.
// A Tensor is a 2-D matrix [rows, cols] plus the recipe that made it.
// Every op in ops.js returns a new Tensor that remembers its parents and a
// small function that pushes gradients back to them. Calling .backward() on a
// scalar walks that graph in reverse (reverse-mode autodiff).

// Float64 keeps the gradient checks in tests/ precise. Weights are saved as float32.
export const DType = Float64Array;

let gradEnabled = true;

/** Run fn without building a graph (use for inference). */
export function noGrad(fn) {
  const prev = gradEnabled;
  gradEnabled = false;
  try {
    return fn();
  } finally {
    gradEnabled = prev;
  }
}

export class Tensor {
  constructor(data, shape) {
    this.data = data;
    this.shape = shape; // [rows, cols]
    this.grad = new DType(data.length);
    this.parents = [];
    this._backward = null;
  }

  get rows() {
    return this.shape[0];
  }
  get cols() {
    return this.shape[1];
  }

  static zeros(rows, cols) {
    return new Tensor(new DType(rows * cols), [rows, cols]);
  }

  static from(array, rows, cols) {
    return new Tensor(DType.from(array), [rows, cols]);
  }

  zeroGrad() {
    this.grad.fill(0);
  }

  item() {
    return this.data[0];
  }

  /** Back-propagate from this scalar (1x1) tensor. Gradients accumulate into .grad. */
  backward() {
    if (this.data.length !== 1) throw new Error('backward() needs a scalar tensor');
    // iterative depth-first topological sort (no recursion, so deep graphs are safe)
    const topo = [];
    const seen = new Set([this]);
    const stack = [[this, 0]];
    while (stack.length) {
      const top = stack[stack.length - 1];
      const [node, i] = top;
      if (i < node.parents.length) {
        top[1]++;
        const p = node.parents[i];
        if (!seen.has(p)) {
          seen.add(p);
          stack.push([p, 0]);
        }
      } else {
        topo.push(node);
        stack.pop();
      }
    }
    this.grad[0] += 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      if (topo[i]._backward) topo[i]._backward();
    }
  }
}

/** Build an op output. `bw(outGrad, out)` must add gradients into the parents. */
export function make(data, shape, parents, bw) {
  const t = new Tensor(data, shape);
  if (gradEnabled) {
    t.parents = parents;
    t._backward = () => bw(t.grad, t);
  }
  return t;
}
