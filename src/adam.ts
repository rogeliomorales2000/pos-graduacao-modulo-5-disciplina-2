import Tensor from "./tensor";
import type { Parameter } from "./types.ts";

export class Adam {
    private t = 0;
    private m = new Map<string, Tensor>();
    private v = new Map<string, Tensor>();

    constructor(
        private lr: number,
        private beta1 = 0.9,
        private beta2 = 0.999,
        private eps = 1e-8
    ) {}

    step(parameters: Parameter[]) {
        this.t++;

        for (const param of parameters) {
            if (!(param.data instanceof Tensor) || !(param.grad instanceof Tensor)) {
                continue;
            }

            const data = param.data;
            const grad = param.grad;
            const prevM = this.m.get(param.name) ?? Tensor.zeros(data.shape);
            const prevV = this.v.get(param.name) ?? Tensor.zeros(data.shape);

            const nextM = new Float64Array(data.data.length);
            const nextV = new Float64Array(data.data.length);

            for (let i = 0; i < data.data.length; i++) {
                const g = grad.data[i];
                nextM[i] = this.beta1 * prevM.data[i] + (1 - this.beta1) * g;
                nextV[i] = this.beta2 * prevV.data[i] + (1 - this.beta2) * g * g;

                const mHat = nextM[i] / (1 - this.beta1 ** this.t);
                const vHat = nextV[i] / (1 - this.beta2 ** this.t);

                data.data[i] -= this.lr * (mHat / (Math.sqrt(vHat) + this.eps));
                grad.data[i] = 0;
            }

            this.m.set(param.name, new Tensor(nextM, [...data.shape]));
            this.v.set(param.name, new Tensor(nextV, [...data.shape]));
        }
    }
}