// src/kernels/ops.ts
// GPU Operations using GPUTensor structures.
// Keeps compute execution purely on the GPU, avoiding unnecessary CPU memory copies.

import { GPUTensor } from "../gpu/tensor";
import { createBuffer } from "../gpu/buffer";
import { createComputePipeline } from "../gpu/pipeline";

import matmulShader from "../shaders/matmul.wgsl?raw";
import softmaxShader from "../shaders/softmax.wgsl?raw";
import rmsnormShader from "../shaders/layernorm.wgsl?raw";
import elementwiseShader from "../shaders/elementwise.wgsl?raw";

// WeakMap to cache compiled GPUComputePipelines by device reference to prevent recompile stutters.
const pipelineCache = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();

function getPipeline(
    device: GPUDevice,
    shaderCode: string,
    entryPoint: string = "main"
): GPUComputePipeline {
    let devCache = pipelineCache.get(device);
    if (!devCache) {
        devCache = new Map<string, GPUComputePipeline>();
        pipelineCache.set(device, devCache);
    }

    const key = `${shaderCode}::${entryPoint}`;
    let pipeline = devCache.get(key);
    if (!pipeline) {
        pipeline = createComputePipeline(device, shaderCode, entryPoint);
        devCache.set(key, pipeline);
    }
    return pipeline;
}

/**
 * Performs matrix multiplication A x B = C on GPUTensors.
 * Shape: A is (M, K), B is (K, N), C is (M, N)
 */
export async function matmul(
    device: GPUDevice,
    a: GPUTensor,
    b: GPUTensor
): Promise<GPUTensor> {
    const [M, K] = a.shape;
    const [K_b, N] = b.shape;

    if (K !== K_b) {
        throw new Error(`Matmul dimension mismatch: A columns (${K}) must match B rows (${K_b}).`);
    }

    // Create output tensor
    const c = new GPUTensor(device, [M, N], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    // Upload dimensions uniform
    const uniformsData = new Uint32Array([M, K, N, 0]);
    const bufferUniforms = createBuffer(
        device,
        new Float32Array(uniformsData.buffer),
        GPUBufferUsage.UNIFORM
    );

    const pipeline = getPipeline(device, matmulShader);
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: a.buffer } },
            { binding: 1, resource: { buffer: b.buffer } },
            { binding: 2, resource: { buffer: c.buffer } },
            { binding: 3, resource: { buffer: bufferUniforms } },
        ],
    });

    const encoder = device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);

    const workgroupsX = Math.ceil(N / 8);
    const workgroupsY = Math.ceil(M / 8);
    computePass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
    computePass.end();

    device.queue.submit([encoder.finish()]);
    bufferUniforms.destroy(); // Free temporary uniforms buffer

    return c;
}

/**
 * Performs softmax normalization on a GPUTensor.
 * Shape: x is (N,)
 */
export async function softmax(
    device: GPUDevice,
    x: GPUTensor
): Promise<GPUTensor> {
    const N = x.numel;
    const output = new GPUTensor(device, x.shape, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    const uniformsData = new Uint32Array([N, 0]);
    const bufferUniforms = createBuffer(
        device,
        new Float32Array(uniformsData.buffer),
        GPUBufferUsage.UNIFORM
    );

    const pipeline = getPipeline(device, softmaxShader);
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: x.buffer } },
            { binding: 1, resource: { buffer: output.buffer } },
            { binding: 2, resource: { buffer: bufferUniforms } },
        ],
    });

    const encoder = device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(1, 1, 1); // Grid stride reduction
    computePass.end();

    device.queue.submit([encoder.finish()]);
    bufferUniforms.destroy();

    return output;
}

/**
 * Performs RMSNorm normalization on x using scaling weights.
 * Shape: x is (D,), weight is (D,)
 */
export async function rmsnorm(
    device: GPUDevice,
    x: GPUTensor,
    weight: GPUTensor,
    epsilon: number = 1e-5
): Promise<GPUTensor> {
    const D = x.numel;
    if (weight.numel !== D) {
        throw new Error(`RMSNorm size mismatch: weight shape must match input tensor elements.`);
    }

    const output = new GPUTensor(device, x.shape, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    const uniformsBufferData = new ArrayBuffer(16);
    const u32View = new Uint32Array(uniformsBufferData);
    const f32View = new Float32Array(uniformsBufferData);
    u32View[0] = D;
    f32View[1] = epsilon;

    const bufferUniforms = createBuffer(
        device,
        new Float32Array(uniformsBufferData),
        GPUBufferUsage.UNIFORM
    );

    const pipeline = getPipeline(device, rmsnormShader);
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: x.buffer } },
            { binding: 1, resource: { buffer: weight.buffer } },
            { binding: 2, resource: { buffer: output.buffer } },
            { binding: 3, resource: { buffer: bufferUniforms } },
        ],
    });

    const encoder = device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(1, 1, 1);
    computePass.end();

    device.queue.submit([encoder.finish()]);
    bufferUniforms.destroy();

    return output;
}

/**
 * Performs elementwise addition: a + b = c
 * Shape: a is (N,), b is (N,)
 */
export async function add(
    device: GPUDevice,
    a: GPUTensor,
    b: GPUTensor
): Promise<GPUTensor> {
    const N = a.numel;
    if (b.numel !== N) {
        throw new Error("Elementwise addition requires matching sizes.");
    }

    const c = new GPUTensor(device, a.shape, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    const uniformsData = new Uint32Array([1, N, 0, 0]); // op = 1 (Add)
    const bufferUniforms = createBuffer(
        device,
        new Float32Array(uniformsData.buffer),
        GPUBufferUsage.UNIFORM
    );

    const pipeline = getPipeline(device, elementwiseShader);
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: a.buffer } },
            { binding: 1, resource: { buffer: b.buffer } },
            { binding: 2, resource: { buffer: c.buffer } },
            { binding: 3, resource: { buffer: bufferUniforms } },
        ],
    });

    const encoder = device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);

    const workgroupSize = 256;
    const dispatchX = Math.ceil(N / workgroupSize);
    computePass.dispatchWorkgroups(dispatchX, 1, 1);
    computePass.end();

    device.queue.submit([encoder.finish()]);
    bufferUniforms.destroy();

    return c;
}

/**
 * Performs SwiGLU merge operation: SiLU(gate) * up = c.
 * Shape: gate is (N,), up is (N,)
 */
export async function silu_mul(
    device: GPUDevice,
    gate: GPUTensor,
    up: GPUTensor
): Promise<GPUTensor> {
    const N = gate.numel;
    if (up.numel !== N) {
        throw new Error("SwiGLU gating requires matching sizes.");
    }

    const c = new GPUTensor(device, gate.shape, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    const uniformsData = new Uint32Array([3, N, 0, 0]); // op = 3 (SiLU_Mul)
    const bufferUniforms = createBuffer(
        device,
        new Float32Array(uniformsData.buffer),
        GPUBufferUsage.UNIFORM
    );

    const pipeline = getPipeline(device, elementwiseShader);
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: gate.buffer } },
            { binding: 1, resource: { buffer: up.buffer } },
            { binding: 2, resource: { buffer: c.buffer } },
            { binding: 3, resource: { buffer: bufferUniforms } },
        ],
    });

    const encoder = device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);

    const workgroupSize = 256;
    const dispatchX = Math.ceil(N / workgroupSize);
    computePass.dispatchWorkgroups(dispatchX, 1, 1);
    computePass.end();

    device.queue.submit([encoder.finish()]);
    bufferUniforms.destroy();

    return c;
}
