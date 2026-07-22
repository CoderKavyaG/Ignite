// src/shaders/layernorm.wgsl
// Root Mean Square Normalization (RMSNorm) compute shader.
// RMSNorm is a simplified, more efficient variant of LayerNorm used in LLaMA-style LLMs (like SmolLM2).
// Formula: x_i' = x_i / sqrt(mean(x²) + epsilon) * weight_i
//
// Tensor Shape Trace:
// - input: Shape (D,), flat Float32Array vector to normalize.
// - weight: Shape (D,), flat Float32Array scaling parameters (gamma).
// - output: Shape (D,), normalized output vector.
// - uniforms: Struct containing [dimension, epsilon, pad, pad2], aligned to 16 bytes.

struct Uniforms {
  dimension: u32,
  epsilon: f32,
  pad: u32,
  pad2: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

// Shared memory for parallel reduction of mean-squared values on-chip.
var<workgroup> shared_mean_sq: f32;
var<workgroup> local_scratch_sq: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let tid = local_id.x;
  let d = uniforms.dimension;
  let eps = uniforms.epsilon;
  const WG_SIZE = 256u;

  // 1. Thread-local accumulation of sum-of-squares using a grid-stride loop.
  // Helps handle vectors that are larger than the workgroup size (256).
  var thread_sq_sum: f32 = 0.0;
  for (var i = tid; i < d; i += WG_SIZE) {
    let val = input[i];
    thread_sq_sum += val * val;
  }
  local_scratch_sq[tid] = thread_sq_sum;
  workgroupBarrier(); // Sync local shared memory writes

  // 2. Parallel reduction in workgroup shared memory
  for (var stride = WG_SIZE / 2u; stride > 0u; stride /= 2u) {
    if (tid < stride) {
      local_scratch_sq[tid] = local_scratch_sq[tid] + local_scratch_sq[tid + stride];
    }
    workgroupBarrier();
  }

  // 3. Thread 0 calculates the mean of squares and cache inverted divisor
  if (tid == 0u) {
    let sum_squares = local_scratch_sq[0];
    let mean_squares = sum_squares / f32(d);
    shared_mean_sq = mean_squares;
  }
  workgroupBarrier();

  let mean_squares = shared_mean_sq;
  let rms_inv = 1.0 / sqrt(mean_squares + eps);

  // 4. Normalize, scale with weights, and write outputs
  for (var i = tid; i < d; i += WG_SIZE) {
    output[i] = input[i] * rms_inv * weight[i];
  }
}
