import type { Window } from "./types";

export function makeWindows(
    tokenIds: number[],
    contextSize: number
): Window[] {
    if (contextSize < 1) {
        throw new Error(`context size must be => 1`)
    }

    const windows: Window[] = []
    for (let i = 0; i + contextSize < tokenIds.length; i++) {
        const slice = tokenIds.slice(i, i + contextSize + 1)
        windows.push({
            input: slice.slice(0, contextSize),
            target: slice.slice(1),
        })
    }

    return windows
}

export default makeWindows
export { makeWindows as createWindows }
