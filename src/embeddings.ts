import Tensor from "./tensor";
import type { Nullable } from "./types.ts";
import { randomMatrix } from "./utils";

export class Embedding {
    lastIds: Nullable<number[]> = null;
    weight: Tensor;
    gradWeight: Nullable<Tensor> = null;

    constructor(vocabSize: number, dim: number, seed: number) {
        this.weight = randomMatrix(vocabSize, dim, seed);
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

    forward(ids: number[]): Tensor {
        this.lastIds = ids;
        const dim = this.weight.shape[1];
        const out = Tensor.zeros([ids.length, dim]);

        for (let t = 0; t < ids.length; t++) {
            const id = ids[t];
            if (id < 0 || id >= this.weight.shape[0]) {
                throw new Error(`Token id ${id} out of range`);
            }
            for (let d = 0; d < dim; d++) {
                out.set(this.weight.get(id, d), t, d);
            }
        }

        return out;
    }

    backward(gradOutput: Tensor): Tensor {
        if (this.lastIds == null) {
            throw new Error("Must call forward() before");
        }

        const dim = this.weight.shape[1];
        this.gradWeight = Tensor.zeros(this.weight.shape);

        for (let t = 0; t < this.lastIds.length; t++) {
            const id = this.lastIds[t];
            for (let d = 0; d < dim; d++) {
                const value = this.gradWeight.get(id, d) + gradOutput.get(t, d);
                this.gradWeight.set(value, id, d);
            }
        }

        return gradOutput.clone();
    }
}
