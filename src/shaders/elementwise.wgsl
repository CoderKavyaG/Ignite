// src/shaders/elementwise.wgsl
// Multi-operation elementwise compute shader for tensor arithmetic & activations.
// Operation types are determined dynamically by the uniform 'op' index:
// - op = 0: SiLU activation -> C[i] = A[i] * sigmoid(A[i])
// - op = 1: Array Add -> C[i] = A[i] + B[i] (used for residual connections)
// - op = 2: Array Multiply -> C[i] = A[i] * B[i] (used for SwiGLU gating)
//
// Tensor Shape Trace:
// - A: Shape (N,), flat Float32Array.
// - B: Shape (N,), flat Float32Array.
// - C: Shape (N,), flat Float32Array (output).
// - uniforms: Struct containing [op, N, padding, padding2] (aligned to 16 bytes).

struct Uniforms {
  op: u32,
  N: u32,
  pad: u32,
  pad2: u32,
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let idx = global_id.x;
  if (idx >= uniforms.N) {
    return;
  }

  let op = uniforms.op;
  if (op == 0u) {
    // SiLU Activation. Formula: x * sigmoid(x) = x / (1.0 + exp(-x)).
    // Numerically stable exp is handled natively by GPUs for standard ranges.
    let x = A[idx];
    C[idx] = x / (1.0 + exp(-x));
  } else if (op == 1u) {
    // Elementwise add (Residual connection)
    C[idx] = A[idx] + B[idx];
  } else if (op == 2u) {
    // Elementwise multiply (SwiGLU gating layer)
    C[idx] = A[idx] * B[idx];
  } else if (op == 3u) {
    // SiLU_Mul (SwiGLU gating merge element): C[i] = SiLU(A[i]) * B[i]
    let gate_val = A[idx];
    let up_val = B[idx];
    C[idx] = (gate_val / (1.0 + exp(-gate_val))) * up_val;
  }
}
