// src/shaders/rope.wgsl
// Rotary Position Embeddings (RoPE) compute shader.
// Rotates the Q or K tensor in-place to encode positional information.
//
// Tensor Shape Trace:
// - Q / K: Shape (seq_len, n_heads, d_head) -> processed flat
// - cos_table: Precomputed float table of shape (max_seq_len, d_head)
// - sin_table: Precomputed float table of shape (max_seq_len, d_head)
// - uniforms: Struct containing [seq_len, n_heads, d_head, pos_offset] (16 bytes)

struct Uniforms {
  seq_len: u32,
  n_heads: u32,
  d_head: u32,
  pos_offset: u32,
};

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> cos_table: array<f32>;
@group(0) @binding(2) var<storage, read> sin_table: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let local_i = global_id.x;
  
  let seq_len = uniforms.seq_len;
  let n_heads = uniforms.n_heads;
  let d_head = uniforms.d_head;
  let half_d = d_head / 2u;
  let pos_offset = uniforms.pos_offset;

  let total_pairs = seq_len * n_heads * half_d;

  // Boundary check
  if (local_i >= total_pairs) {
    return;
  }

  // Decompose 1D thread ID into sequence, head, and dimension coordinates
  let pair_idx = local_i % half_d;
  let head_idx = (local_i / half_d) % n_heads;
  let seq_idx = local_i / (n_heads * half_d);

  // Flat indices for coords (2*i, 2*i + 1) in x
  let offset = seq_idx * (n_heads * d_head) + head_idx * d_head;
  let idx0 = offset + (2u * pair_idx);
  let idx1 = idx0 + 1u;

  // Global position index of the token in the wider sequence (for precomputed tables lookup)
  let token_pos = seq_idx + pos_offset;

  // Cos/Sin table index for target position and head-dimension pair (2*i)
  let table_idx = token_pos * d_head + (2u * pair_idx);

  let cos_val = cos_table[table_idx];
  let sin_val = sin_table[table_idx];

  let v0 = x[idx0];
  let v1 = x[idx1];

  // Apply matrix rotation:
  // [v0_rot] = [ cos(angle) -sin(angle) ] [v0]
  // [v1_rot]   [ sin(angle)  cos(angle) ] [v1]
  x[idx0] = v0 * cos_val - v1 * sin_val;
  x[idx1] = v0 * sin_val + v1 * cos_val;
}
