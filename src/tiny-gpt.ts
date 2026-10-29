import { RELU } from "./activation";
import { CasualSelfAttention } from "./attention";
import { CrossEntropyLoss } from "./cel";
import { Dense } from "./dense";
import { Embedding } from "./embeddings";
import { PositionalEmbedding } from "./positional-embedding";
import { Tensor } from "./tensor";
import type { Parameter } from "./types.ts";
import { randomMatrix } from "./utils";

export type TinyGptConfig = {
  vocabSize: number;
  dim: number;
  contextSize: number;
  hiddenMul: number;
  seed: number;
};

export function defaultGptConfig(vocabSize: number): TinyGptConfig {
  return {
    vocabSize,
    dim: 32,
    contextSize: 64,
    hiddenMul: 2,
    seed: 42,
  }
}

export class TransformerBlock {
  attn: CasualSelfAttention;
  ffn1: Dense; // expandir: dim -> dim * hiddenMul
  relu: RELU;
  ffn2: Dense; // comprime de volta: dim * hiddenMul -> dim

  constructor(dim: number, hiddenMul: number, seed: number) {
    this.attn = new CasualSelfAttention(dim, seed);

    this.ffn1 = new Dense(
      randomMatrix(dim, dim * hiddenMul, seed + 10),
      Tensor.zeros([dim * hiddenMul]),
    )
    this.ffn2 = new Dense(
      randomMatrix(
        dim * hiddenMul,
        dim,
        seed + 11
      ),
      Tensor.zeros([dim]),
    )

    this.relu = new RELU();
  }

  forward(x: Tensor): Tensor {
    const a = this.attn.forward(x);
    // residual: atalho x + attn(x) - gradiente não some
    const h = x.add(a);

    // FFN: Dense -> ReLU -> Dense (não-lineariedade local por posição)
    const f = this.ffn2.forward(
      this.relu.forward(
        this.ffn1.forward(h)
      )
    )

    // segundo residual: h + ffn(h)
    return h.add(f);
  }

  backward(gradOutput: Tensor): Tensor {
    const gradFfn2 = this.ffn2.backward(gradOutput);
    const gradRelu = this.relu.backward(gradFfn2);
    const gradFfn1 = this.ffn1.backward(gradRelu);
    const gradH = gradOutput.add(gradFfn1);

    const gradAttn = this.attn.backward(gradH);
    return gradH.add(gradAttn);
  }

  parameters(prefix = "block"): Parameter[] {
    // Empacota attn + FFN com nomes prefixados para o Adam
    return [
      ...this.attn.parameters().map((p) => ({ ...p, name: `${prefix}.${p.name}` })),
      {
        name: `${prefix}.ffn1.w`,
        data: this.ffn1.weights,
        grad: this.ffn1.gradWeights!,
      },
      { name: `${prefix}.ffn1.b`, data: this.ffn1.bias, grad: this.ffn1.gradBias! },
      {
        name: `${prefix}.ffn2.w`,
        data: this.ffn2.weights,
        grad: this.ffn2.gradWeights!,
      },
      { name: `${prefix}.ffn2.b`, data: this.ffn2.bias, grad: this.ffn2.gradBias! },
    ];
  }
}

export class TinyGPT {
  config: TinyGptConfig;
  tokenEmb: Embedding;
  posEmb: PositionalEmbedding;
  block: TransformerBlock;
  lmHead: Dense;
  lossFn = new CrossEntropyLoss();
  lastLoss = 0;
  trace: { name: string, shape: number[] }[] = []

  constructor(config: TinyGptConfig) {
    this.config = config;
    this.tokenEmb = new Embedding(
      config.vocabSize,
      config.dim,
      config.seed
    )
    this.posEmb = new PositionalEmbedding(
      config.contextSize,
      config.dim,
      config.seed + 1
    )
    this.block = new TransformerBlock(
      config.dim,
      config.hiddenMul,
      config.seed + 2
    )
    this.lmHead = new Dense(
      randomMatrix(config.dim, config.vocabSize, config.seed + 3),
      Tensor.zeros([config.vocabSize]),
    )
  }

  forward(ids: number[], recordTrace = false): Tensor {
    if (ids.length === 0) {
      throw new Error("TinyGPT.forward needs at least one token.")
    }

    if (ids.length > this.config.contextSize) {
      throw new Error(
        `Sequence length ${ids.length} > context ${this.config.contextSize}`
      )
    }

    this.trace = [];

    // ID -> vetor de token
    const tok = this.tokenEmb.forward(ids)
    if (recordTrace) {
      this.trace.push({
        name: "token emb", shape: [...tok.shape]
      });
    }

    // Soma posição (ordem da sequência)
    const x0 = this.posEmb.forward(tok)
    if (recordTrace) {
      this.trace.push({
        name: "+ position", shape: [...x0.shape]
      });
    }

    // Bloco transformer (atenção + FFN + residuais)
    const x1 = this.block.forward(x0)
    if (recordTrace) {
      this.trace.push({
        name: "transformer block", shape: [...x1.shape]
      });
    }

    // Logits sobre o vocabulário
    const logits = this.lmHead.forward(x1)
    if (recordTrace) {
      this.trace.push({
        name: "lm head logits", shape: [...logits.shape]
      });
    }

    return logits;
  }

  backward(logits: Tensor, targets: number[]): void {
    this.lastLoss = this.lossFn.forward(logits, targets);

    let grad = this.lossFn.backward();
    grad = this.lmHead.backward(grad);
    grad = this.block.backward(grad);
    grad = this.posEmb.backward(grad);
    this.tokenEmb.backward(grad);
  }

  parameters(): Parameter[] {
    // Tudo que o Adam precisa atualizar
    return [
      {
        name: "tokenEmb",
        data: this.tokenEmb.weight,
        grad: this.tokenEmb.gradWeight!,
      },
      {
        name: "posEmb",
        data: this.posEmb.weight,
        grad: this.posEmb.gradWeight!,
      },
      ...this.block.parameters("block0"),
      {
        name: "lmHead.w",
        data: this.lmHead.weights,
        grad: this.lmHead.gradWeights!,
      },
      { name: "lmHead.b", data: this.lmHead.bias, grad: this.lmHead.gradBias! },
    ];
  }

  summary(): string {
    const params = this.parameters().reduce((n, p) => {
      const size = p.data instanceof Tensor ? p.data.data.length : 1;
      return n + size;
    }, 0);
    return [
      "TinyGPT",
      `  vocab:    ${this.config.vocabSize}`,
      `  dim:      ${this.config.dim}`,
      `  context:  ${this.config.contextSize}`,
      `  blocks:   1 (single-head attention + FFN)`,
      `  params:   ${params}`,
    ].join("\n");
  }
}