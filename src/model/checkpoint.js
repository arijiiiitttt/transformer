// Save / load a model as two plain files:
//   config.json   architecture, vocabulary, task, training progress, parameter layout
//   weights.bin   all parameters as raw float32, back to back
import fs from 'node:fs';
import path from 'node:path';
import { Transformer } from './transformer.js';

export function saveModel(model, dir, meta = {}) {
  fs.mkdirSync(dir, { recursive: true });
  const params = model.parameters();
  const layout = [];
  let total = 0;
  for (const [name, t] of params) {
    layout.push({ name, shape: t.shape, offset: total });
    total += t.data.length;
  }
  const buf = new Float32Array(total);
  params.forEach(([, t], i) => buf.set(t.data, layout[i].offset));
  fs.writeFileSync(path.join(dir, 'weights.bin'), Buffer.from(buf.buffer));
  const config = { model: model.cfg, parameters: layout, ...meta };
  fs.writeFileSync(path.join(dir, 'config.json'), JSON.stringify(config, null, 2));
}

export function loadModel(dir) {
  const config = JSON.parse(fs.readFileSync(path.join(dir, 'config.json'), 'utf8'));
  const raw = fs.readFileSync(path.join(dir, 'weights.bin'));
  const weights = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  const model = new Transformer(config.model);
  const byName = new Map(model.parameters());
  for (const { name, shape, offset } of config.parameters) {
    const t = byName.get(name);
    if (!t) throw new Error(`checkpoint has unknown parameter ${name}`);
    if (t.shape[0] !== shape[0] || t.shape[1] !== shape[1]) throw new Error(`shape mismatch for ${name}`);
    t.data.set(weights.subarray(offset, offset + t.data.length));
  }
  return { model, config };
}

export function checkpointExists(dir) {
  return fs.existsSync(path.join(dir, 'config.json')) && fs.existsSync(path.join(dir, 'weights.bin'));
}
