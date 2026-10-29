import { TextDataset } from "./text-dataset";
import { defaultGptConfig, TinyGPT } from "./tiny-gpt";
import {
  trainGpt,
  generate,
} from "./train-gpt";

const corpus = `User: hi
Bot: hello there
User: how are you
Bot: i am well
User: bye
Bot: see you
`;

const contextSize = 32;
const ds = TextDataset.fromText(corpus, contextSize);
console.log(ds.describe());
console.log("");

const model = new TinyGPT({
  ...defaultGptConfig(ds.tokenizer.vocabSize),
  contextSize,
  dim: 32,
});
console.log(model.summary());
console.log("\nQuick overfit on the toy dialogue…\n");

trainGpt(model, ds.windows("train"), {
  epochs: 40,
  learningRate: 0.01,
  logEvery: 10,
});

const sample = generate(model, ds.tokenizer, "User: hi\nBot:", {
  maxNewTokens: 40,
  temperature: 0.5,
});
console.log("\nSample generation:");
console.log("----------");
console.log(sample);
console.log("----------");

console.log("\nNext: ship train / chat / inspect CLIs — the mini ChatGPT product.");