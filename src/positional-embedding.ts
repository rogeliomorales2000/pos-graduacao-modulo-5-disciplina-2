import Tensor from "./tensor";
import type { Nullable } from "./types.ts";
import { randomMatrix } from "./utils";

export class PositionalEmbedding {
    lastInput: Nullable<Tensor> = null;
    weight: Tensor;
    gradWeight: Nullable<Tensor> = null;

    constructor(contextSize: number, dim: number, seed: number) {
        this.weight = randomMatrix(contextSize, dim, seed);
    }

    get weights(): Tensor {
        return this.weight;
    }

    set weights(value: Tensor) {
        this.weight = value;
    }

    get gradWeights(): Nullable<Tensor> {
        return this.gradWeight;
    }

    set gradWeights(value: Nullable<Tensor>) {
        this.gradWeight = value;
    }

    forward(input: Tensor): Tensor {
        if (input.rank !== 2) {
            throw new Error("PositionalEmbedding expects [T, dim]");
        }

        this.lastInput = input.clone();
        const [seqLen, dim] = input.shape as [number, number];
        const out = Tensor.zeros(input.shape);

        for (let t = 0; t < seqLen; t++) {
            for (let d = 0; d < dim; d++) {
                const value = input.get(t, d) + this.weight.get(t, d);
                out.set(value, t, d);
            }
        }

        return out;
    }

    backward(gradOutput: Tensor): Tensor {
        if (this.lastInput == null) {
            throw new Error("Must call forward() before");
        }

        const [seqLen, dim] = this.lastInput.shape as [number, number];
        this.gradWeight = Tensor.zeros(this.weight.shape);

        for (let t = 0; t < seqLen; t++) {
            for (let d = 0; d < dim; d++) {
                const value = this.gradWeight.get(t, d) + gradOutput.get(t, d);
                this.gradWeight.set(value, t, d);
            }
        }

        return gradOutput.clone();
    }
}
