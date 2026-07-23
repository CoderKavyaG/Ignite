// src/model/config.ts
// Configuration parameters for the SmolLM2-135M model architecture.

export const Config = {
    n_layers: 30,
    d_model: 576,
    n_heads: 9,
    d_head: 64,
    n_kv_heads: 3, // Grouped Query Attention (GQA) factor: 3 Query heads per 1 KV head (9 / 3 = 3)
    d_ffn: 1536,   // Hidden intermediate dimension of the FFN/SwiGLU block
    vocab_size: 49152,
    max_seq_len: 2048,
    rope_base: 10000,
    rms_norm_eps: 1e-5,
} as const;
