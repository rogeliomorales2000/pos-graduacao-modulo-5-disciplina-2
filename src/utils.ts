import Tensor from "./tensor";

export function randomMatrix(rows: number, cols: number, seed: number): Tensor {
    let state = seed;
    const data: number[] = [];

    for (let i = 0; i < rows * cols; i++) {
        state = (state * 1664525 + 1013904223) % 4294967296;
        const u = state / 4294967296;
        data.push((u - 0.5) * 0.5);
    }

    return new Tensor(new Float64Array(data), [rows, cols]);
}


export function shuffle<T>(arr: T[], seed: number): T[] {
    const out = [...arr];
    let state = seed || 1;

    for (let i = out.length - 1; i > 0; i--) {
        state = (state * 1664525 + 1013904223) % 4294967296;
        const j = state % (i + 1);
        const tmp = out[i];
        out[i] = out[j];
        out[j] = tmp;
    }

    return out;
}
