import { createBuffer } from "../gpu/buffer";
import { createComputePipeline } from "../gpu/pipeline";
import rmsnormShader from "../shaders/layernorm.wgsl?raw";

/**
 * Computes Layer Normalization (specifically RMSNorm) over a 1D Float32Array vector.
 *
 * Tensor Shape Trace:
 * - input: Shape (D,), flat Float32Array vector to normalize.
 * - weight: Shape (D,), flat Float32Array scaling parameters (gamma).
 * - output: Shape (D,), flat Float32Array normalized result.
 * - Uniforms: Shape (4) -> [D, epsilon, padding, padding2] (aligned to 16 bytes).
 *
 * @param device The WebGPU GPUDevice context.
 * @param x Input vector.
 * @param weight Scaling weight parameters.
 * @param epsilon Tiny offset to prevent division-by-zero. Defaults to 1e-5.
 */
export async function rmsnorm(
    device: GPUDevice,
    x: Float32Array,
    weight: Float32Array,
    epsilon: number = 1e-5
): Promise<Float32Array> {
    const D = x.length;
    if (D === 0) {
        throw new Error("Cannot run RMSNorm on empty vector.");
    }
    if (weight.length !== D) {
        throw new Error(`RMSNorm size mismatch: input size (${D}) must match weight size (${weight.length}).`);
    }

    // 1. Setup buffers
    const bufferInput = createBuffer(device, x, GPUBufferUsage.STORAGE);
    const bufferWeight = createBuffer(device, weight, GPUBufferUsage.STORAGE);

    const byteLength = D * Float32Array.BYTES_PER_ELEMENT;
    const bufferOutput = createBuffer(
        device,
        byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    const uniformsBufferData = new ArrayBuffer(16);
    const u32View = new Uint32Array(uniformsBufferData);
    const f32View = new Float32Array(uniformsBufferData);
    u32View[0] = D;
    f32View[1] = epsilon; // Epsilon is standard Float32

    const bufferUniforms = createBuffer(
        device,
        new Float32Array(uniformsBufferData),
        GPUBufferUsage.UNIFORM
    );

    const bufferStaging = device.createBuffer({
        size: byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // 2. Setup pipeline and bind group
    const pipeline = createComputePipeline(device, rmsnormShader);

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: bufferInput } },
            { binding: 1, resource: { buffer: bufferWeight } },
            { binding: 2, resource: { buffer: bufferOutput } },
            { binding: 3, resource: { buffer: bufferUniforms } },
        ],
    });

    // 3. Dispatch and execute compute pass
    const encoder = device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);

    // Single workgroup execution for local-level reductions
    computePass.dispatchWorkgroups(1, 1, 1);
    computePass.end();

    encoder.copyBufferToBuffer(bufferOutput, 0, bufferStaging, 0, byteLength);
    device.queue.submit([encoder.finish()]);

    // 4. Map staging buffer and fetch results
    await bufferStaging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(bufferStaging.getMappedRange().slice(0));

    // 5. Clean up
    bufferStaging.unmap();
    bufferInput.destroy();
    bufferWeight.destroy();
    bufferOutput.destroy();
    bufferUniforms.destroy();
    bufferStaging.destroy();

    return result;
}
