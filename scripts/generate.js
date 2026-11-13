// Run a trained model.
//   node scripts/generate.js "123456"
//   node scripts/generate.js "123456" --dir model --temp 0.7
import { CharTokenizer } from '../src/data/tokenizer.js';
import { generateText } from '../src/generate/decode.js';
import { loadModel } from '../src/model/checkpoint.js';
import { mulberry32 } from '../src/utils/rng.js';

const args = process.argv.slice(2);
let dir = 'model';
let temperature = 0;
let seed = Date.now();
const inputs = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--dir') dir = args[++i];
  else if (args[i] === '--temp') temperature = Number(args[++i]);
  else if (args[i] === '--seed') seed = Number(args[++i]);
  else inputs.push(args[i]);
}
if (inputs.length === 0) {
  console.error('usage: node scripts/generate.js "<input text>" [--dir model] [--temp 0] [--seed N]');
  process.exit(1);
}

const { model, config } = loadModel(dir);
const tokenizer = new CharTokenizer(config.alphabet);
const rng = mulberry32(seed);
console.log(`task: ${config.task}   alphabet: ${config.alphabet}   trained steps: ${config.stepsDone}`);
for (const text of inputs) {
  const out = generateText(model, tokenizer, text, { maxNew: model.cfg.maxLen - 1, temperature, rng });
  console.log(`${text} -> ${out}`);
}
