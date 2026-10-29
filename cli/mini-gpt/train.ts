/**
 * CLI: npm run train:mini-gpt
 *
 * Next-token on documents. Index is only for optional related-doc display.
 */

import path from "node:path";
import ps from "node:process";
import { trainMiniGpt, loadModel } from "../../src/mini-gpt";
import {
  buildDocStore,
  loadKnowledgeDir,
  saveDocStore,
} from "../../src/rag";

const root = ps.cwd();

const epochsArg = process.argv.find((a) => a.startsWith("--epochs="));
const epochs = epochsArg ? Number(epochsArg.split("=")[1]) : 60;
const modelPath = path.join(root, "models", "mini-gpt.json");

trainMiniGpt({
  datasetRoot: path.join(root, "dataset", "mini-gpt"),
  modelPath,
  epochs,
  learningRate: 0.004,
  contextSize: 96,
  dim: 64,
});

const { model, tokenizer } = loadModel(modelPath);
const chunks = loadKnowledgeDir(path.join(root, "dataset", "mini-gpt","knowledge"));
const store = buildDocStore(chunks, model.tokenEmb, tokenizer);
const storePath = path.join(root, "models", "doc-store.json");
saveDocStore(store, storePath);
console.log(
  `\nIndexed ${store.chunks.length} knowledge chunks → ${storePath}`,
);