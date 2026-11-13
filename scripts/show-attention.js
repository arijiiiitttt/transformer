// Print what the model attends to. Shows the cross-attention of the last decoder
// layer (decoder positions down, source positions across), averaged over heads.
//   node scripts/show-attention.js "12345"
import { noGrad } from '../src/autograd/tensor.js';
import { CharTokenizer } from '../src/data/tokenizer.js';
import { generateIds } from '../src/generate/decode.js';
import { BOS } from '../src/masks/masks.js';
import { loadModel } from '../src/model/checkpoint.js';

const text = process.argv[2];
const dir = process.argv[3] ?? 'model';
if (!text) {
  console.error('usage: node scripts/show-attention.js "<input text>" [model dir]');
  process.exit(1);
}
const { model, config } = loadModel(dir);
const tok = new CharTokenizer(config.alphabet);
const src = tok.encode(text);
const { ids } = generateIds(model, src, { maxNew: model.cfg.maxLen - 1 });

noGrad(() => {
  const decIn = [BOS, ...ids];
  model.decode(decIn, model.encode(src), src);
  const layer = model.decoder.layers[model.decoder.layers.length - 1];
  const maps = layer.crossAttn.lastWeights;
  const ns = src.length;
  console.log(`\ninput : ${text}\noutput: ${tok.decode(ids)}\n`);
  console.log('cross-attention, last decoder layer, mean over heads');
  console.log('        ' + [...text].map((c) => `  ${c}  `).join(''));
  for (let i = 0; i < decIn.length; i++) {
    let row = '';
    for (let j = 0; j < ns; j++) {
      const w = maps.reduce((s, m) => s + m.data[i * ns + j], 0) / maps.length;
      row += ' ' + w.toFixed(2) + ' ';
    }
    const label = i === 0 ? 'BOS' : tok.decode([ids[i - 1]]);
    console.log(`  in:${label.padEnd(3)} ${row}`);
  }
});
