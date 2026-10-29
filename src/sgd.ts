import Tensor from "./tensor";
import type { Parameter } from "./types.ts";

export class SGD {
    constructor(private learningrate: number) {}

    step(params: Parameter[]) {
        for (const param of params) {
            if (typeof param.data === "number" && typeof param.grad === "number") {
                param.data -= this.learningrate * param.grad;
                param.grad = 0;
                continue;
            }

            if (param.data instanceof Tensor && param.grad instanceof Tensor) {
                for (let i = 0; i < param.data.data.length; i++) {
                    param.data.data[i] -= this.learningrate * param.grad.data[i];
                    param.grad.data[i] = 0;
                }
            }
        }
    }
}

