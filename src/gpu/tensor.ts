// src/gpu/tensor.ts
// GPUTensor: A lightweight tensor class wrapping GPUBuffer and tracking shape metadata.
// Minimizes Host-Device (CPU-GPU) transfer overheads by keeping data residing in VRAM between ops.

export class GPUTensor {
    device: GPUDevice;
    buffer: GPUBuffer;
    shape: number[];
    dtype: 'f32' = 'f32';

    /**
     * Initializes a new GPUTensor. If an existing GPUBuffer is not supplied, a new one is allocated.
     */
    constructor(
        device: GPUDevice,
        shape: number[],
        usage?: GPUBufferUsageFlags,
        existingBuffer?: GPUBuffer
    ) {
        this.device = device;
        this.shape = [...shape];

        if (existingBuffer) {
            this.buffer = existingBuffer;
        } else {
            const sizeBytes = this.byteSize;

            // Default to read/write storage + copy operations capability
            const resolvedUsage = usage ?? (
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_SRC |
                GPUBufferUsage.COPY_DST
            );

            // Force standard WebGPU 4-byte buffer alignment
            const alignedBytes = Math.ceil(sizeBytes / 4) * 4;

            this.buffer = device.createBuffer({
                size: alignedBytes,
                usage: resolvedUsage,
            });
        }
    }

    /**
     * Instantiates a GPUTensor by copying a Float32Array from host (CPU) memory to global device memory (GPU VRAM).
     */
    static fromFloat32Array(
        device: GPUDevice,
        data: Float32Array,
        shape: number[]
    ): GPUTensor {
        const requiredNumel = shape.reduce((acc, dim) => acc * dim, 1);
        if (data.length !== requiredNumel) {
            throw new Error(
                `Tensor shape mismatch: cannot shape Float32Array of size ${data.length} into shape [${shape.join(', ')}] ` +
                `(expected elements: ${requiredNumel})`
            );
        }

        // Allocate storage buffer with COPY_DST to enable queue writes
        const tensor = new GPUTensor(
            device,
            shape,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );

        device.queue.writeBuffer(
            tensor.buffer,
            0,
            data.buffer,
            data.byteOffset,
            data.byteLength
        );

        return tensor;
    }

    /**
     * Total number of elements in the tensor.
     */
    get numel(): number {
        if (this.shape.length === 0) return 0;
        return this.shape.reduce((acc, dim) => acc * dim, 1);
    }

    /**
     * Total byte size of the tensor (number of elements * Float32 byte scale).
     */
    get byteSize(): number {
        return this.numel * Float32Array.BYTES_PER_ELEMENT;
    }

    /**
     * Asynchronously reads data residing in GPU memory back to a fresh Float32Array on the CPU.
     * Leverages a short-lived staging buffer to map VRAM memory back to host space.
     */
    async toFloat32Array(): Promise<Float32Array> {
        const sizeBytes = this.byteSize;
        if (sizeBytes === 0) {
            return new Float32Array(0);
        }

        const staging = this.device.createBuffer({
            size: Math.ceil(sizeBytes / 4) * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.buffer, 0, staging, 0, sizeBytes);
        this.device.queue.submit([encoder.finish()]);

        await staging.mapAsync(GPUMapMode.READ);
        const mappedRange = staging.getMappedRange();
        const result = new Float32Array(mappedRange.slice(0));

        // Release staging buffer immediately
        staging.unmap();
        staging.destroy();

        return result;
    }

    /**
     * Manually destroys and releases the underlying GPUBuffer memory allocations.
     */
    destroy(): void {
        this.buffer.destroy();
    }
}
