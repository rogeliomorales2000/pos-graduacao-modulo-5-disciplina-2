export class CharTokenizer {
    readonly stoi : Map<string,number>;
    readonly itos : Map<number,string>
    constructor(vocab:string[]){
        this.stoi = new Map()
        this.itos = new Map()

        for (let i = 0; i < vocab.length; i++) {
            const ch = vocab[i]
            if (this.stoi.has(ch)) {
                throw new Error(`Duplicate vocab entry: ${JSON.stringify(ch, null, 2)}`)
            }
            this.stoi.set(ch, i)
            this.itos.set(i, ch)
        }
    }

    static fromText(text:string): CharTokenizer{
        const unique = [...new Set(text.split(""))].sort()
        return new CharTokenizer(unique)
    }

    static fromJSON(data: { vocab: string[] }): CharTokenizer {
        return new CharTokenizer(data.vocab);
    }

    toJSON(): { vocab: string[] } {
        return {
            vocab: Array.from(this.stoi.entries())
                .sort((a, b) => a[1] - b[1])
                .map(([token]) => token),
        };
    }

    get vocabSize(): number {
        return this.stoi.size;
    }

    encode (
        textp : string
    ): number[]{
        const ids: number[] = []

        for (const c of textp){
            const id = this.stoi.get(c)
            if (id == undefined){
                throw new Error(`Unknow Char ${JSON.stringify(c)}`)

            }
            ids.push(id)
        }
        return ids
    }
    decode (ids : number[]):string{
        return ids.map((id) => {
            const c = this.itos.get(id)
            if (c == undefined){
                throw new Error(`Unknown token id ${id}`)

            }
            return c
        }).join("")

    }
}