// src/shaders/softmax.wgsl
// Numerically stable softmax compute shader using workgroup shared memory.
//
// Tensor Shape Trace:
// - input: 1D array of shape (N,)
// - output: 1D array of shape (N,)
// - uniforms: shape (2) -> [N, pad], total 8 bytes (aligned to 16 bytes internally)

struct Uniforms {
  N: u32,
  pad: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

// Workgroup size of 256 threads.
// Shared memory buffers:
// - var<workgroup> is local on-chip SRAM, shared ONLY among threads within the SAME workgroup.
// - It is extremely low latency compared to storage buffers (global VRAM), which are off-chip and slow.
// - We use it to perform low-cost parallel reductions (max value and sum of exponentials).
var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;
var<workgroup> local_scratch_max: array<f32, 256>;
var<workgroup> local_scratch_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let tid = local_id.x;
  let N = uniforms.N;
  const WG_SIZE = 256u;

  // -------------------------------------------------------------
  // Pass 1: Find the maximum value in the vector (for numerical stability)
  // -------------------------------------------------------------
  // Step 1.1: Each thread computes its local max using a grid-stride loop
  var thread_max: f32 = -3.402823466e+38f; // -FLT_MAX (minimum f32 value)
  for (var i = tid; i < N; i += WG_SIZE) {
    thread_max = max(thread_max, input[i]);
  }
  local_scratch_max[tid] = thread_max;
  workgroupBarrier(); // Synchronize shared memory writes

  // Step 1.2: Perform leaf-reduction on shared memory
  for (var stride = WG_SIZE / 2u; stride > 0u; stride /= 2u) {
    if (tid < stride) {
      local_scratch_max[tid] = max(local_scratch_max[tid], local_scratch_max[tid + stride]);
    }
    workgroupBarrier();
  }

  // Step 1.3: Thread 0 broadcasts the global max to workgroup shared memory
  if (tid == 0u) {
    shared_max = local_scratch_max[0];
  }
  workgroupBarrier();
  
  let global_max = shared_max;

  // -------------------------------------------------------------
  // Pass 2: Calculate exponentials and their total sum
  // -------------------------------------------------------------
  // Step 2.1: Thread-local sum of exp(val - max)
  var thread_sum: f32 = 0.0;
  for (var i = tid; i < N; i += WG_SIZE) {
    let val = exp(input[i] - global_max);
    thread_sum += val;
  }
  local_scratch_sum[tid] = thread_sum;
  workgroupBarrier();

  // Step 2.2: Leaf-reduction for sum of exponentials
  for (var stride = WG_SIZE / 2u; stride > 0u; stride /= 2u) {
    if (tid < stride) {
      local_scratch_sum[tid] = local_scratch_sum[tid] + local_scratch_sum[tid + stride];
    }
    workgroupBarrier();
  }

  // Step 2.3: Thread 0 broadcasts global sum to shared memory
  if (tid == 0u) {
    shared_sum = local_scratch_sum[0];
  }
  workgroupBarrier();

  let global_sum = shared_sum;

  // -------------------------------------------------------------
  // Pass 3: Divide elementwise exponentials by the global sum
  // -------------------------------------------------------------
  for (var i = tid; i < N; i += WG_SIZE) {
    output[i] = exp(input[i] - global_max) / global_sum;
  }
}
