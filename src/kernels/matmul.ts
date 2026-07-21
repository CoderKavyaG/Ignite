import { createBuffer } from "../gpu/buffer";
import { createComputePipeline } from "../gpu/pipeline";
import matmulShader from "../shaders/matmul.wgsl?raw";

/**
 * Performs matrix multiplication C = A x B on the GPU.
 *
 * Tensor Shape Trace:
 * - A: Input matrix, shape (M, K), flat size M * K elements.
 * - B: Input matrix, shape (K, N), flat size K * N elements.
 * - C: Output matrix, shape (M, N), flat size M * N elements.
 * - Uniforms: Shape (4) containing [M, K, N, padding], size 16 bytes.
 *
 * @param device The WebGPU GPUDevice context.
 * @param A Flat Float32Array containing input matrix A.
 * @param B Flat Float32Array containing input matrix B.
 * @param M Number of rows in A / rows in C.
 * @param K Number of columns in A / rows in B.
 * @param N Number of columns in B / columns in C.
 */
export async function matmul(
    device: GPUDevice,
    A: Float32Array,
    B: Float32Array,
    M: number,
    K: number,
    N: number
): Promise<Float32Array> {
    // 1. Validate inputs to prevent out-of-bounds access or GPU crashes
    if (A.length !== M * K) {
        throw new Error(`Matrix A shape mismatch: expected size ${M * K} (for ${M}x${K}), got ${A.length}.`);
    }
    if (B.length !== K * N) {
        throw new Error(`Matrix B shape mismatch: expected size ${K * N} (for ${K}x${N}), got ${B.length}.`);
    }

    // 2. Create GPU buffers and upload data
    // Input Buffer A: shape (M, K), usage GPUBufferUsage.STORAGE (read-only in shader)
    const bufferA = createBuffer(
        device,
        A,
        GPUBufferUsage.STORAGE
    );

    // Input Buffer B: shape (K, N), usage GPUBufferUsage.STORAGE (read-only in shader)
    const bufferB = createBuffer(
        device,
        B,
        GPUBufferUsage.STORAGE
    );

    // Output Buffer C: shape (M, N), size M * N * 4 bytes.
    // Must be STORAGE to be written to by the compute shader, and COPY_SRC so we can copy its final state to a staging buffer.
    const sizeC = M * N * Float32Array.BYTES_PER_ELEMENT;
    const bufferC = createBuffer(
        device,
        sizeC,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    // Uniform Buffer: shape (4) representing [M, K, N, pad].
    // Stored as Uint32Array (4-byte unsigned integers) to align with WGSL u32 variables.
    const uniformData = new Uint32Array([M, K, N, 0]);
    const bufferUniforms = createBuffer(
        device,
        new Float32Array(uniformData.buffer), // Safe cast to Float32Array for our createBuffer helper, matching internal byte length
        GPUBufferUsage.UNIFORM
    );

    // Staging Buffer: shape (M, N), size M * N * 4 bytes.
    // Must support MAP_READ (so CPU can examine it) and COPY_DST (so GPU can write to it from bufferC).
    const bufferStaging = device.createBuffer({
        size: sizeC,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // 3. Compile WGSL and construct the compute pipeline
    const pipeline = createComputePipeline(device, matmulShader);

    // 4. Create the bind group to map our resources to the layout slots defined in WGSL (@group(0) @binding(x))
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: bufferA } },
            { binding: 1, resource: { buffer: bufferB } },
            { binding: 2, resource: { buffer: bufferC } },
            { binding: 3, resource: { buffer: bufferUniforms } },
        ],
    });

    // 5. Encode the compute commands
    const encoder = device.createCommandEncoder();
    const computePass = encoder.beginComputePass();

    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);

    // Dispatch workgroups to cover all elements in C of shape (M, N).
    // Workgroup size is 8x8, meaning each workgroup handles an 8x8 block of output values.
    const workgroupsX = Math.ceil(N / 8);
    const workgroupsY = Math.ceil(M / 8);

    computePass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
    computePass.end();

    // 6. Copy GPU computed buffer C into CPU-readable Staging Buffer
    encoder.copyBufferToBuffer(bufferC, 0, bufferStaging, 0, sizeC);

    // 7. Submit commands to the GPU command queue
    device.queue.submit([encoder.finish()]);

    // 8. Map staging buffer asynchronously and copy back contents to CPU memory
    // This is a asynchronous operation since CPU must wait for GPU to complete the execution queues.
    await bufferStaging.mapAsync(GPUMapMode.READ);
    const mappedRange = bufferStaging.getMappedRange();

    // Clone the mapped array buffer so it remains in JS memory after the staging buffer gets unmapped
    const result = new Float32Array(mappedRange.slice(0));

    // Clean up GPU allocations to avoid memory leaks
    bufferStaging.unmap();
    bufferA.destroy();
    bufferB.destroy();
    bufferC.destroy();
    bufferUniforms.destroy();
    bufferStaging.destroy();

    return result;
}
