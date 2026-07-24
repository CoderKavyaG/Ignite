// src/tokenizer/tokenizer.ts
// SentencePiece/BPE Tokenizer implemented from scratch.
// Conforms to SmolLM2 LLaMA-style tokenization without external dependencies.

export class Tokenizer {
    private vocab: Map<string, number>;
    private reverseVocab: Map<number, string>;
    private merges: Map<string, number>;

    constructor(
        vocab: Map<string, number>,
        reverseVocab: Map<number, string>,
        merges: Map<string, number>
    ) {
        this.vocab = vocab;
        this.reverseVocab = reverseVocab;
        this.merges = merges;
    }

    /**
     * Instantiates a Tokenizer from standard HuggingFace tokenizer.json configuration object.
     */
    static async fromJSON(tokenizerJson: any): Promise<Tokenizer> {
        const vocab = new Map<string, number>();
        const reverseVocab = new Map<number, string>();
        const merges = new Map<string, number>();

        // Support both raw format and standard HF structure wrapped in model
        const model = tokenizerJson.model || tokenizerJson;
        const vocabObj = model.vocab;
        const mergesArr = model.merges;

        if (!vocabObj) {
            throw new Error("Invalid tokenizer JSON: vocab object not found.");
        }

        for (const [token, id] of Object.entries(vocabObj)) {
            vocab.set(token, id as number);
            reverseVocab.set(id as number, token);
        }

        if (mergesArr) {
            for (let i = 0; i < mergesArr.length; i++) {
                merges.set(mergesArr[i], i);
            }
        }

        return new Tokenizer(vocab, reverseVocab, merges);
    }

    /**
     * Encodes text into an array of token IDs.
     */
    encode(text: string): number[] {
        if (text.length === 0) return [];

        // 1. Pre-tokenization: Split on spaces and preserve them as 'Ġ' prefixes
        // e.g. "hello world" -> ["hello", "Ġworld"]
        const segments: string[] = [];
        let currentWord = "";
        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            if (char === " ") {
                if (currentWord.length > 0) {
                    segments.push(currentWord);
                }
                currentWord = "Ġ";
            } else {
                currentWord += char;
            }
        }
        if (currentWord.length > 0) {
            segments.push(currentWord);
        }

        const finalTokenIds: number[] = [];

        // 2. Perform Byte Pair Encoding (BPE) merges per segment
        for (const segment of segments) {
            let subwords = Array.from(segment);

            while (subwords.length > 1) {
                let bestPairIndex = -1;
                let bestRank = Infinity;

                // Find the adjacent pair with the highest merge priority (lowest rank)
                for (let i = 0; i < subwords.length - 1; i++) {
                    const pair = `${subwords[i]} ${subwords[i + 1]}`;
                    const rank = this.merges.get(pair);
                    if (rank !== undefined && rank < bestRank) {
                        bestRank = rank;
                        bestPairIndex = i;
                    }
                }

                if (bestPairIndex === -1) {
                    break; // No more merges possible for this segment
                }

                // Merge target pair
                const mergedToken = subwords[bestPairIndex] + subwords[bestPairIndex + 1];
                subwords.splice(bestPairIndex, 2, mergedToken);
            }

            // 3. Map subwords to Vocab IDs (handling Byte Fallbacks)
            for (const token of subwords) {
                if (this.vocab.has(token)) {
                    finalTokenIds.push(this.vocab.get(token)!);
                } else {
                    // SentencePiece Byte Fallback: convert unknown chars to hex UTF-8 notation e.g., <0xXX>
                    const bytes = new TextEncoder().encode(token);
                    for (const byte of bytes) {
                        const byteHex = `<0x${byte.toString(16).toUpperCase()}>`;
                        if (this.vocab.has(byteHex)) {
                            finalTokenIds.push(this.vocab.get(byteHex)!);
                        } else {
                            finalTokenIds.push(0); // Fallback to <unk> token (usually ID 0 or 2 depending on tokenizer)
                        }
                    }
                }
            }
        }

        return finalTokenIds;
    }

    /**
     * Decodes an array of token IDs back into readable text.
     */
    decode(ids: number[]): string {
        let result = "";

        for (const id of ids) {
            const token = this.reverseVocab.get(id);
            if (!token) continue;

            // Skip special/added tokens (like <s>, </s>, <unk>, <paragraph>)
            if (token.startsWith("<") && token.endsWith(">") && token.length > 2) {
                continue;
            }

            result += token;
        }

        // Replace both 'Ġ' and the SentencePiece space character ' ' (U+2581) with standard spaces
        const decoded = result
            .replace(/Ġ/g, " ")
            .replace(/\u2581/g, " ");

        return decoded;
    }
}
