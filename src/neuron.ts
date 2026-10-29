import Tensor from "./tensor";
import { sigmoid } from "./types.ts";

export class Neuron {
    constructor(
        public weigths: number[],
        public bias: number
    ) {}

    weightedSum(inputs: number[] | Tensor): number {
        const values = inputs instanceof Tensor ? Array.from(inputs.data) : inputs;
        let sum = this.bias;
        for (let i = 0; i < values.length; i++) {
            sum += values[i] * this.weigths[i];
        }
        return sum;
    }

    forward(inputs: number[] | Tensor, activation?: (x: number) => number) {
        const fn = activation ?? sigmoid;
        return fn(this.weightedSum(inputs));
    }
}