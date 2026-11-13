// Train (or resume training) a Transformer on a toy sequence task.
//   node scripts/train.js --task reverse --steps 1500
import { performance } from 'node:perf_hooks';
import { CharTokenizer } from '../src/data/tokenizer.js';
import { makeDataset, makeSample, TASK_NAMES } from '../src/data/tasks.js';
import { generateText } from '../src/generate/decode.js';
import { BOS, EOS } from '../src/masks/masks.js';
import { checkpointExists, loadModel, saveModel } from '../src/model/checkpoint.js';
import { Transformer } from '../src/model/transformer.js';
import { Adam, clipGradNorm, lrSchedule } from '../src/optim/adam.js';
import { noGrad } from '../src/autograd/tensor.js';
import { mulberry32 } from '../src/utils/rng.js';

const opt = parseArgs(process.argv.slice(2), {
  task: 'reverse',
  alphabet: '0123456789',
  steps: 1500,
  B: 16,
  D: 64,
  L: 2,
  H: 4,
  ff: 128,
  lr: 0.003,
  warmup: 150,
  clip: 1.0,
  minLen: 2,
  maxLen: 8,
  evalEvery: 100,
  evalSamples: 100,
  dir: 'model',
  seed: 1,
  fresh: false,
});

if (!TASK_NAMES.includes(opt.task)) {
  console.error(`unknown task "${opt.task}". Choose from: ${TASK_NAMES.join(', ')}`);
  process.exit(1);
}

// ---- model: resume if a checkpoint exists, otherwise start fresh --------------------------
let model;
let tokenizer;
let stepsDone = 0;
if (!opt.fresh && checkpointExists(opt.dir)) {
  const loaded = loadModel(opt.dir);
  model = loaded.model;
  tokenizer = new CharTokenizer(loaded.config.alphabet);
  stepsDone = loaded.config.stepsDone ?? 0;
  opt.task = loaded.config.task;
  opt.alphabet = loaded.config.alphabet;
  console.log(`resuming from ${opt.dir}/ at step ${stepsDone} (task: ${opt.task})`);
} else {
  tokenizer = new CharTokenizer(opt.alphabet);
  model = new Transformer(
    { vocabSize: tokenizer.vocabSize, dModel: opt.D, heads: opt.H, layers: opt.L, dFF: opt.ff, maxLen: opt.maxLen + 2 },
    opt.seed,
  );
}
if (opt.maxLen + 2 > model.cfg.maxLen) {
  console.error(`--maxLen ${opt.maxLen} needs model maxLen >= ${opt.maxLen + 2}, but this model has ${model.cfg.maxLen}`);
  process.exit(1);
}
console.log(
  `task=${opt.task}  params=${model.countParameters().toLocaleString()}  ` +
    `D=${model.cfg.dModel} layers=${model.cfg.layers} heads=${model.cfg.heads} ff=${model.cfg.dFF}`,
);

// ---- data ----------------------------------------------------------------------------------
const rng = mulberry32(opt.seed * 7919 + stepsDone);
const valSet = makeDataset(opt.task, opt.alphabet, opt.minLen, opt.maxLen, opt.evalSamples, 424242);

function toIds(sample) {
  const src = tokenizer.encode(sample.src);
  const tgt = tokenizer.encode(sample.tgt);
  return { src, tgtIn: [BOS, ...tgt], tgtOut: [...tgt, EOS] };
}

function evaluate() {
  return noGrad(() => {
    let loss = 0;
    let exact = 0;
    for (const s of valSet) {
      const { src, tgtIn, tgtOut } = toIds(s);
      loss += model.loss(src, tgtIn, tgtOut).item();
      if (generateText(model, tokenizer, s.src, { maxNew: opt.maxLen + 2 }) === s.tgt) exact++;
    }
    return { loss: loss / valSet.length, acc: exact / valSet.length };
  });
}

// ---- train ---------------------------------------------------------------------------------
const params = model.parameters();
const adam = new Adam(params);
const t0 = performance.now();
let runLoss = 0;
let runCount = 0;
let tokens = 0;

for (let i = 1; i <= opt.steps; i++) {
  model.zeroGrad();
  for (let b = 0; b < opt.B; b++) {
    const { src, tgtIn, tgtOut } = toIds(makeSample(opt.task, opt.alphabet, opt.minLen, opt.maxLen, rng));
    const loss = model.loss(src, tgtIn, tgtOut);
    loss.backward();
    runLoss += loss.item();
    runCount++;
    tokens += src.length + tgtIn.length;
  }
  const gradScale = 1 / opt.B;
  clipGradNorm(params, opt.clip, gradScale);
  adam.step(lrSchedule(stepsDone + i, opt.lr, opt.warmup), gradScale);

  if (i % opt.evalEvery === 0 || i === opt.steps) {
    const v = evaluate();
    const secs = (performance.now() - t0) / 1000;
    console.log(
      `step ${String(stepsDone + i).padStart(5)} | train loss ${(runLoss / runCount).toFixed(4)} | ` +
        `val loss ${v.loss.toFixed(4)} | exact match ${(v.acc * 100).toFixed(0)}% | ` +
        `${Math.round(tokens / secs)} tok/s | ${secs.toFixed(0)}s`,
    );
    runLoss = 0;
    runCount = 0;
    saveModel(model, opt.dir, {
      task: opt.task,
      alphabet: opt.alphabet,
      stepsDone: stepsDone + i,
      valLoss: v.loss,
      exactMatch: v.acc,
    });
  }
}

console.log(`saved to ${opt.dir}/`);
console.log('\nsamples:');
for (const s of makeDataset(opt.task, opt.alphabet, opt.minLen, opt.maxLen, 5, 99)) {
  const out = generateText(model, tokenizer, s.src, { maxNew: opt.maxLen + 2 });
  console.log(`  ${s.src.padEnd(10)} -> ${out.padEnd(10)} ${out === s.tgt ? 'ok' : `(wanted ${s.tgt})`}`);
}

function parseArgs(argv, defaults) {
  const out = { ...defaults };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const key = a.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    if (!(key in defaults)) {
      console.error(`unknown option ${a}`);
      process.exit(1);
    }
    if (typeof defaults[key] === 'boolean') out[key] = true;
    else out[key] = typeof defaults[key] === 'number' ? Number(argv[++i]) : argv[++i];
  }
  return out;
}
