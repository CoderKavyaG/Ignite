// src/kernels/rope.ts
// Rotary Position Embeddings (RoPE) TypeScript host executor and tables precomputation.

import { GPUTensor } from "../gpu/tensor";
import { createBuffer } from "../gpu/buffer";
import { createComputePipeline } from "../gpu/pipeline";
import ropeShader from "../shaders/rope.wgsl?raw";

/**
 * Precomputes the static Cosine and Sine angle projection tables for RoPE on the CPU.
 * 
 * Tensor Shape Trace:
 * - cosTable / sinTable: Shape (maxSeqLen, dHead), flat Float32Array. Each position's values 
 *   are aligned directly with query / key head dimensions to avoid shader-side index mismatch.
 *
 * @param maxSeqLen Maximum sequence context length (defaults to 2048 for SmolLM2).
 * @param dHead Vector dimension per attention head (defaults to 64 for SmolLM2).
 * @param base Frequency base constant (defaults to 10000).
 */
export function precomputeRoPETables(
    maxSeqLen: number = 2048,
    dHead: number = 64,
    base: number = 10000
): { cosTable: Float32Array; sinTable: Float32Array } {
    const cosTable = new Float32Array(maxSeqLen * dHead);
    const sinTable = new Float32Array(maxSeqLen * dHead);

    for (let pos = 0; pos < maxSeqLen; pos++) {
        for (let dim = 0; dim < dHead; dim++) {
            // Dimensions are grouped in pairs (2*i, 2*i + 1). Both coordinates share the same theta.
            const i = Math.floor(dim / 2) * 2;
            const theta = 1.0 / Math.pow(base, i / dHead);
            const angle = pos * theta;

            cosTable[pos * dHead + dim] = Math.cos(angle);
            sinTable[pos * dHead + dim] = Math.sin(angle);
        }
    }

    return { cosTable, sinTable };
}

/**
 * Applies Rotary Position Embeddings in-place to query or key tensors on the GPU.
 *
 * Tensor Shape Trace:
 * - x: GPUTensor of shape [seqLen, nHeads, dHead] (normally Q or K).
 * - cosTable: GPUTensor of shape [maxSeqLen, dHead] (read-only cached buffer).
 * - sinTable: GPUTensor of shape [maxSeqLen, dHead] (read-only cached buffer).
 * - posOffset: Integer index representing the prompt length offsets in autoregressive loops.
 *
 * @param device The WebGPU GPUDevice context.
 * @param x The tensor to alter with position-angle rotation in-place.
 * @param cosTable Precomputed cosine values.
 * @param sinTable Precomputed sine values.
 * @param posOffset Global context token placement position offset.
 */
export async function rope(
    device: GPUDevice,
    x: GPUTensor,
    cosTable: GPUTensor,
    sinTable: GPUTensor,
    posOffset: number
): Promise<void> {
    const [seqLen, nHeads, dHead] = x.shape;
    const halfD = Math.floor(dHead / 2);
    const totalPairs = seqLen * nHeads * halfD;

    if (totalPairs === 0) return;

    // 1. Setup uniforms containing dimensions and offsets
    const uniformsData = new Uint32Array([seqLen, nHeads, dHead, posOffset]);
    const bufferUniforms = createBuffer(
        device,
        new Float32Array(uniformsData.buffer),
        GPUBufferUsage.UNIFORM
    );

    // 2. Setup pipeline and bind group options
    const pipeline = createComputePipeline(device, ropeShader);
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: x.buffer } },
            { binding: 1, resource: { buffer: cosTable.buffer } },
            { binding: 2, resource: { buffer: sinTable.buffer } },
            { binding: 3, resource: { buffer: bufferUniforms } },
        ],
    });

    // 3. Dispatch kernel
    const encoder = device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);

    const workgroupSize = 256;
    const dispatchX = Math.ceil(totalPairs / workgroupSize);
    computePass.dispatchWorkgroups(dispatchX, 1, 1);
    computePass.end();

    // 4. Submit and cleanup staging uniforms
    device.queue.submit([encoder.finish()]);
    bufferUniforms.destroy();
}
