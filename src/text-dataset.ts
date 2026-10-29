import { makeWindows } from "./make-windown";
import { CharTokenizer } from "./tokenizer";

export class TextDataset {
  constructor(
    public readonly tokenizer: CharTokenizer,
    public readonly trainIds: number[],
    public readonly valIds: number[],
    public readonly testIds: number[],
    public readonly contextSize: number,
  ) {}

  static fromText(
    text: string,
    contextSize: number,
    fractions = { train: 0.8, val: 0.1, test: 0.1 },
  ): TextDataset {
    const tokenizer = CharTokenizer.fromText(text);
    const ids = tokenizer.encode(text);
    const n = ids.length;
    const nTrain = Math.max(contextSize + 1, Math.floor(n * fractions.train));
    const nVal = Math.max(contextSize + 1, Math.floor(n * fractions.val));
    const trainIds = ids.slice(0, nTrain);
    const valIds = ids.slice(nTrain, nTrain + nVal);
    const testIds = ids.slice(nTrain + nVal);
    return new TextDataset(tokenizer, trainIds, valIds, testIds, contextSize);
  }

  static fromSplits(
    splits: { train: string; validation: string; test: string },
    contextSize: number,
  ): TextDataset {
    const all = `${splits.train}\n${splits.validation}\n${splits.test}`;
    const tokenizer = CharTokenizer.fromText(all);
    return new TextDataset(
      tokenizer,
      tokenizer.encode(splits.train),
      tokenizer.encode(splits.validation),
      tokenizer.encode(splits.test),
      contextSize,
    );
  }

  windows(split: "train" | "validation" | "test") {
    const ids =
      split === "train"
        ? this.trainIds
        : split === "validation"
          ? this.valIds
          : this.testIds;
    return makeWindows(ids, this.contextSize);
  }

  describe(): string {
    return [
      "TextDataset",
      `  vocab:     ${this.tokenizer.vocabSize}`,
      `  train ids: ${this.trainIds.length}`,
      `  val ids:   ${this.valIds.length}`,
      `  test ids:  ${this.testIds.length}`,
      `  context:   ${this.contextSize}`,
      `  train wins:${this.windows("train").length}`,
    ].join("\n");
  }
}
