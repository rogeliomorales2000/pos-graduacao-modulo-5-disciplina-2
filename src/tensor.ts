export class Tensor {
    constructor(
        public data: Float64Array,
        public shape: number[]
    ) {}

    static fromMatrix(matrix: number[][]): Tensor {
        const rows = matrix.length;
        const cols = matrix[0]?.length ?? 0;
        const data: number[] = [];

        for (const row of matrix) {
            if (row.length !== cols) {
                throw new Error("All rows must have the same length");
            }
            data.push(...row);
        }

        return new Tensor(new Float64Array(data), [rows, cols]);
    }

    static zeros(shape: number[]): Tensor {
        const size = shape.reduce((a, b) => a * b, 1);
        return new Tensor(new Float64Array(size), shape);
    }

    get(...indices: number[]): number {
        let idx = 0;
        let stride = 1;

        for (let i = this.shape.length - 1; i >= 0; i--) {
            idx += indices[i] * stride;
            stride *= this.shape[i];
        }

        return this.data[idx];
    }

    set(value: number, ...indices: number[]) {
        let idx = 0;
        let stride = 1;

        for (let d = this.shape.length - 1; d >= 0; d--) {
            idx += indices[d] * stride;
            stride *= this.shape[d];
        }

        this.data[idx] = value;
    }

    map(fn: (value: number, index: number) => number): Tensor {
        const result = new Float64Array(this.data.length);

        for (let i = 0; i < this.data.length; i++) {
            result[i] = fn(this.data[i], i);
        }

        return new Tensor(result, [...this.shape]);
    }

    reshape(newShape: number[]): Tensor {
        const expected = newShape.reduce((a, b) => a * b, 1);
        if (expected !== this.data.length) {
            throw new Error(
                `Cannot reshape [${this.shape.join(", ")}] to [${newShape.join(", ")}] : size mismatch.`
            );
        }
        return new Tensor(new Float64Array(this.data), newShape);
    }

    clone(): Tensor {
        return new Tensor(new Float64Array(this.data), [...this.shape]);
    }

    get rank(): number {
        return this.shape.length;
    }

    matmul(other: Tensor): Tensor {
        if (this.rank !== 2 || other.rank !== 2) {
            throw new Error("matmul requires two 2D tensors (matrices)");
        }

        const [m, n] = this.shape as [number, number];
        const [n2, p] = other.shape as [number, number];

        if (n !== n2) {
            throw new Error(
                `Cannot matmul [${m}, ${n}] with [${n2}, ${p}]: inner dims must match.`
            );
        }

        const result = Tensor.zeros([m, p]);

        for (let i = 0; i < m; i++) {
            for (let j = 0; j < p; j++) {
                let sum = 0;

                for (let k = 0; k < n; k++) {
                    sum += this.get(i, k) * other.get(k, j);
                }

                result.set(sum, i, j);
            }
        }

        return result;
    }

    transpose(): Tensor {
        if (this.rank !== 2) {
            throw new Error("transpose currently supports 2D tensors only.");
        }

        const [rows, cols] = this.shape as [number, number];
        const result = Tensor.zeros([cols, rows]);

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                result.set(this.get(r, c), c, r);
            }
        }

        return result;
    }

    scale(factor: number): Tensor {
        return new Tensor(
            new Float64Array(this.data.map((v) => v * factor)),
            [...this.shape]
        );
    }

    add(other: Tensor): Tensor {
        this.assertSameShape(other);

        const data = new Float64Array(this.data.length);
        for (let i = 0; i < this.data.length; i++) {
            data[i] = this.data[i] + other.data[i];
        }

        return new Tensor(data, [...this.shape]);
    }

    assertSameShape(other: Tensor) {
        if (
            this.shape.length !== other.shape.length ||
            this.shape.some((s, i) => s !== other.shape[i])
        ) {
            throw new Error(
                `Shape mismatch: [${this.shape.join(", ")}] vs [${other.shape.join(", ")}]`
            );
        }
    }

    toArray(): number[] {
        return Array.from(this.data);
    }

    toString(): string {
        return `Tensor(${JSON.stringify(this.toArray())})`;
    }
}

export default Tensor;



