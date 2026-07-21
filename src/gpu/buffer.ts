/**
 * Helper to create a GPUBuffer and optionally write data to it.
 * 
 * Tensor shape comment:
 * - A Float32Array represents flat elements of a tensor, with byteLength = length * 4.
 *
 * @param device The WebGPU device singleton.
 * @param dataOrSize A Float32Array containing initialized data to upload, OR a number indicating size in bytes.
 * @param usage Flag combination representing the intended GPU buffer usages (e.g. GPUBufferUsage.STORAGE, COPY_SRC, etc.)
 */
export function createBuffer(
    device: GPUDevice,
    dataOrSize: Float32Array | number,
    usage: GPUBufferUsageFlags
): GPUBuffer {
    const size = typeof dataOrSize === "number" ? dataOrSize : dataOrSize.byteLength;

    // Enforce standard WGSL 4-byte boundaries alignment. Since each Float32 element is 4 bytes,
    // this is naturally aligned for Float32Array, but this safeguard ensures we never violate alignment constraints.
    const alignedSize = Math.ceil(size / 4) * 4;

    // If initial data is provided, the buffer must have COPY_DST usage for writeBuffer to succeed
    const resolvedUsage = typeof dataOrSize !== "number"
        ? (usage | GPUBufferUsage.COPY_DST)
        : usage;

    const buffer = device.createBuffer({
        size: alignedSize,
        usage: resolvedUsage,
    });

    // If the user provided initial data, copy it using queue.writeBuffer.
    // This is highly optimal as WebGPU handles staging memory allocation under the hood.
    if (typeof dataOrSize !== "number") {
        device.queue.writeBuffer(
            buffer,
            0,
            dataOrSize.buffer,
            dataOrSize.byteOffset,
            dataOrSize.byteLength
        );
    }

    return buffer;
}
