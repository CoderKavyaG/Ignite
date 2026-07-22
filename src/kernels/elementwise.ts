import { createBuffer } from "../gpu/buffer";
import { createComputePipeline } from "../gpu/pipeline";
import elementwiseShader from "../shaders/elementwise.wgsl?raw";

/**
 * Executes elementwise operations (SiLU, Add, or Multiply) on the GPU.
 *
 * Tensor Shape Trace:
 * - A: Shape (N,), flat Float32Array.
 * - B: Shape (N,) or dummy (1,) if unused, flat Float32Array.
 * - C: Shape (N,), flat Float32Array outputs.
 * - Uniforms: Shape (4) -> [op, N, padding, padding2] (aligned to 16 bytes).
 *
 * @param device The WebGPU GPUDevice context.
 * @param a First input vector (Shape N).
 * @param b Second input vector (Shape N), or null if executing unary SiLU.
 * @param op Operation selector: 0 = SiLU, 1 = Add, 2 = Multiply.
 */
export async function elementwise(
    device: GPUDevice,
    a: Float32Array,
    b: Float32Array | null,
    op: number
): Promise<Float32Array> {
    const N = a.length;
    if (N === 0) {
        throw new Error("Cannot run elementwise operations on empty vector.");
    }

    // If binary operation (Add/Multiply), verify B's dimensions.
    if (op !== 0) {
        if (!b) {
            throw new Error(`Operation op=${op} requires a valid tensor B.`);
        }
        if (b.length !== N) {
            throw new Error(`Size mismatch for binary elementwise operation: A size (${N}) and B size (${b.length}) must match.`);
        }
    }

    const byteLength = N * Float32Array.BYTES_PER_ELEMENT;

    // 1. Create buffers.
    const bufferA = createBuffer(device, a, GPUBufferUsage.STORAGE);

    let bufferB: GPUBuffer;
    if (b) {
        bufferB = createBuffer(device, b, GPUBufferUsage.STORAGE);
    } else {
        // WebGPU Gotcha: We cannot leave binding slots empty in active bind groups, even if the shader
        // branch does not read them. We initialize a dummy 4-byte buffer to avoid runtime exceptions.
        bufferB = createBuffer(device, new Float32Array([0.0]), GPUBufferUsage.STORAGE);
    }

    const bufferOutput = createBuffer(
        device,
        byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    const uniformsData = new Uint32Array([op, N, 0, 0]);
    const bufferUniforms = createBuffer(
        device,
        new Float32Array(uniformsData.buffer),
        GPUBufferUsage.UNIFORM
    );

    const bufferStaging = device.createBuffer({
        size: byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // 2. Setup pipeline and bind group
    const pipeline = createComputePipeline(device, elementwiseShader);

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: bufferA } },
            { binding: 1, resource: { buffer: bufferB } },
            { binding: 2, resource: { buffer: bufferOutput } },
            { binding: 3, resource: { buffer: bufferUniforms } },
        ],
    });

    // 3. Encode & Submit.
    // We use workgroup size of 256, dispatching enough workgroups to cover N elements.
    const workgroupSize = 256;
    const dispatchX = Math.ceil(N / workgroupSize);

    const encoder = device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(dispatchX, 1, 1);
    computePass.end();

    encoder.copyBufferToBuffer(bufferOutput, 0, bufferStaging, 0, byteLength);
    device.queue.submit([encoder.finish()]);

    // 4. Map & readback staging buffer.
    await bufferStaging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(bufferStaging.getMappedRange().slice(0));

    // 5. Clean up.
    bufferStaging.unmap();
    bufferA.destroy();
    bufferB.destroy();
    bufferOutput.destroy();
    bufferUniforms.destroy();
    bufferStaging.destroy();

    return result;
}
