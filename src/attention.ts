import { Dense } from "./dense";
import { Tensor } from "./tensor";
import type { Nullable, Parameter } from "./types.ts";
import { randomMatrix } from "./utils";

const NEG_INF = -1e9;

// Transformar scores brutos numa distribuição de probabilidade
export class Softmax {
  private lastOutput: Nullable<Tensor> = null;

  // Scores brutos -> probabilidade (cada linha soma 1)
  forward(logits: Tensor): Tensor {
    if (logits.rank !== 2) {
      throw new Error("Softmax expcets a 2D tenser [rows, cols]")
    }

    const [rows, cols] = logits.shape as [number, number];
    const out = Tensor.zeros([rows, cols]);

    // Precisamos achar o maior score de linha porque exp(1000) estoure o float
    for (let r = 0; r < rows; r++) {
      let max = -Infinity; // Começa abaixo de tudo para o primeiro valor ganhar

      for (let c = 0; c < cols; c++) {
        max = Math.max(max, logits.get(r, c));
      }

      let sum = 0;
      for (let c = 0; c < cols; c++) {
        // Deixar tudo positivo (probabilidade não pode ser negativa)
        const e = Math.exp(logits.get(r, c) - max);
        out.set(e, r, c);
        sum += e;
      }

      // Dividir pela soma -> Virar uma probabilidade
      // Agora cada colunar/celula esta exclusivamente entre [0, 1]
      // E a linha inteira soma ~1
      for (let c = 0; c < cols; c++) {
        out.set(out.get(r, c) / sum, r, c);
      }
    }

    this.lastOutput = out;
    return out;
  }

  backward(gradOutput: Tensor): Tensor {
    if (this.lastOutput === null) {
      throw new Error("Must call forward() before backward().");
    }

    const y = this.lastOutput;
    const [rows, cols] = y.shape as [number, number];
    const grad = Tensor.zeros([rows, cols]);

    for (let r = 0; r < rows; r++) {
      let dot = 0;

      for (let c = 0; c < cols; c++) {
        dot += gradOutput.get(r, c) * y.get(r, c);
      }

      for (let c = 0; c < cols; c++) {
        grad.set(y.get(r, c) * (gradOutput.get(r, c) - dot), r, c);
      }
    }

    return grad;
  }
}

export function applyCasualMask(
  scores: Tensor
): Tensor {
  const T = scores.shape[0];

  if (scores.shape[1] !== T) {
    throw new Error("Casual mask expects square [T, T] scores.")
  }

  const output = scores.clone();

  for (let q = 0; q < T; q++) {
    for (let k = 0; k < T; k++) {
      if (k > q) {
        output.set(NEG_INF, q, k);
      }
    }
  }

  return output;
}

export class CasualSelfAttention {
  wq: Dense; // projeção query = "o que eu estou procurando?"
  wk: Dense; // projeção key = "o que cada token oferece>"
  wv: Dense; // projeção value = "a informação que entrega"
  wo: Dense; // projeção de saída
  softmax = new Softmax();
  scale: number;

  lastWeights: Nullable<Tensor> = null;
  lastQ: Nullable<Tensor> = null;
  lastK: Nullable<Tensor> = null;
  lastV: Nullable<Tensor> = null;
  lastO: Nullable<Tensor> = null;

  constructor(dim: number, seed = 42) {
    this.wq = new Dense(
      randomMatrix(dim, dim, seed), Tensor.zeros([dim])
    );
    this.wk = new Dense(
      randomMatrix(dim, dim, seed + 1), Tensor.zeros([dim])
    );
    this.wv = new Dense(
      randomMatrix(dim, dim, seed + 1), Tensor.zeros([dim])
    );
    this.wo = new Dense(
      randomMatrix(dim, dim, seed + 1), Tensor.zeros([dim])
    );

    // escala clássica do paper
    this.scale = 1 / Math.sqrt(dim)
  }

  forward(x: Tensor): Tensor {
    if (x.rank !== 2) {
      throw new Error("CsualSelfAttention expects [T, dim]");
    }

    const q = this.wq.forward(x);
    const k = this.wk.forward(x);
    const v = this.wv.forward(x);

    // similaridade entre posições
    let scores = q.matmul(k.transpose()).scale(this.scale);
    scores = applyCasualMask(scores); // bloqueia o futuro
    const weights = this.softmax.forward(scores); // vira distrução
    const mixed = weights.matmul(v); // mistura ponderada dos valores
    const out = this.wo.forward(mixed); // projeção final

    this.lastQ = q;
    this.lastK = k;
    this.lastV = v;
    this.lastWeights = weights;
    this.lastO = out;

    return out;
  }

  backward(gradOutput: Tensor): Tensor {
    if (
      this.lastK === null ||
      this.lastO === null ||
      this.lastQ === null ||
      this.lastV === null ||
      this.lastWeights === null
    ) {
      throw new Error("Must call forward() before backward().")
    }

    let gradMixed = this.wo.backward(gradOutput);
    const gradV = this.lastWeights.transpose().matmul(gradMixed);
    const gradWeights = gradMixed.matmul(this.lastV.transpose());

    let gradScores = this.softmax.backward(gradWeights);
    gradScores = gradScores.scale(this.scale);

    const gradQ = gradScores.matmul(this.lastK);
    const gradK = gradScores.transpose().matmul(this.lastQ);

    const gq = this.wq.backward(gradQ);
    const gk = this.wk.backward(gradK);
    const gv = this.wv.backward(gradV);

    return gq.add(gk).add(gv);
  }

  // Lista parâmetros nomeados para o otimizador (Adam)
  parameters(): Parameter[] {
    return [
      { name: "attn.wq.w", data: this.wq.weights, grad: this.wq.gradWeights! },
      { name: "attn.wq.b", data: this.wq.bias, grad: this.wq.gradBias! },
      { name: "attn.wk.w", data: this.wk.weights, grad: this.wk.gradWeights! },
      { name: "attn.wk.b", data: this.wk.bias, grad: this.wk.gradBias! },
      { name: "attn.wv.w", data: this.wv.weights, grad: this.wv.gradWeights! },
      { name: "attn.wv.b", data: this.wv.bias, grad: this.wv.gradBias! },
      { name: "attn.wo.w", data: this.wo.weights, grad: this.wo.gradWeights! },
      { name: "attn.wo.b", data: this.wo.bias, grad: this.wo.gradBias! },
    ];
  }
}