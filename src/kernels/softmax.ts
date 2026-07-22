import { createBuffer } from "../gpu/buffer";
import { createComputePipeline } from "../gpu/pipeline";
import softmaxShader from "../shaders/softmax.wgsl?raw";

/**
 * Computes the numerically stable Softmax over a 1D Float32Array vector on the GPU.
 *
 * Tensor Shape Trace:
 * - input: Shape (N,), flat Float32Array.
 * - output: Shape (N,), flat Float32Array.
 * - Uniforms: [N, pad], Float32Array representing 2 uint32 elements (8 bytes).
 *
 * @param device The WebGPU GPUDevice context.
 * @param data Input vector.
 */
export async function softmax(
    device: GPUDevice,
    data: Float32Array
): Promise<Float32Array> {
    const N = data.length;
    if (N === 0) {
        throw new Error("Cannot compute softmax on an empty vector.");
    }

    // 1. Create buffers
    const bufferInput = createBuffer(device, data, GPUBufferUsage.STORAGE);

    const byteLength = N * Float32Array.BYTES_PER_ELEMENT;
    const bufferOutput = createBuffer(
        device,
        byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    const uniformsData = new Uint32Array([N, 0]);
    const bufferUniforms = createBuffer(
        device,
        new Float32Array(uniformsData.buffer),
        GPUBufferUsage.UNIFORM
    );

    const bufferStaging = device.createBuffer({
        size: byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // 2. Compile pipeline
    const pipeline = createComputePipeline(device, softmaxShader);

    // 3. Create Bind Group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: bufferInput } },
            { binding: 1, resource: { buffer: bufferOutput } },
            { binding: 2, resource: { buffer: bufferUniforms } },
        ],
    });

    // 4. Encode and submit
    const encoder = device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);

    // We dispatch exactly 1 workgroup of 256 threads.
    // The shader executes a grid-stride loop inside this single workgroup
    // to perform workgroup-level reductions for max and sum of exps.
    computePass.dispatchWorkgroups(1, 1, 1);
    computePass.end();

    encoder.copyBufferToBuffer(bufferOutput, 0, bufferStaging, 0, byteLength);
    device.queue.submit([encoder.finish()]);

    // 5. Read back
    await bufferStaging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(bufferStaging.getMappedRange().slice(0));

    // 6. Clean up
    bufferStaging.unmap();
    bufferInput.destroy();
    bufferOutput.destroy();
    bufferUniforms.destroy();
    bufferStaging.destroy();

    return result;
}
