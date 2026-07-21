// Matmul.wgsl: Matrix multiplication compute shader.
// Tensors shape layout:
// - matrixA: MxK storage buffer, accessed flat [row * K + k]
// - matrixB: KxN storage buffer, accessed flat [k * N + col]
// - matrixC: MxN output storage buffer, accessed flat [row * N + col]

// Uniforms struct containing matrix dimensions.
// WebGPU uniform buffers require 16-byte size alignment block constraints on many GPU targets.
// With three u32 variables (M, K, N = 12 bytes), we add a 'pad' field to achieve exactly 16 bytes.
struct Uniforms {
  M: u32,
  K: u32,
  N: u32,
  pad: u32,
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

// We partition the execution space into 2D workgroups of size 8x8.
//
// Why @workgroup_size(8, 8)?
// 1. Hardware Occupancy: Most modern GPU hardware executes threads in units of 32 (Nvidia Warps)
//    or 64 (AMD Wavefronts). A workgroup size of 8 * 8 = 64 matches this vector scheduling block perfectly,
//    preventing half-empty warps/wavefronts and ensuring high hardware utilization.
// 2. 2D Layout Mapping: A 2x2 workgroup dimension aligns naturally with the rows and columns of matrices,
//    making index calculation and spatial tiling logic clean and readable.
// 3. Register Limits: Large workgroup sizes (like 16x16 = 256 threads) can bottleneck on register allocation
//    per multiprocessor, while a moderate 8x8 size leaves ample registers for each thread to perform computations
//    without spilling to slow device memory.
@compute @workgroup_size(8, 8)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let col = global_id.x; // Column offset of C (maps to N)
  let row = global_id.y; // Row offset of C (maps to M)

  // Boundary check: A thread may map outside matrix bounds if M or N is not a multiple of 8.
  if (row >= uniforms.M || col >= uniforms.N) {
    return;
  }

  // Accumulate the dot product of A[row, :] and B[:, col]
  // Tensor Shape Trace:
  // - A[row * K + k] -> shape (M, K)
  // - B[k * N + col] -> shape (K, N)
  var acc: f32 = 0.0;
  let K = uniforms.K;
  
  for (var k: u32 = 0u; k < K; k = k + 1u) {
    let indexA = row * K + k;
    let indexB = k * uniforms.N + col;
    acc = acc + A[indexA] * B[indexB];
  }

  // Write the accumulated dot product out to C[row, col]
  // - C[row * N + col] -> shape (M, N)
  let indexC = row * uniforms.N + col;
  C[indexC] = acc;
}
