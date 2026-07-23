// src/model/loader.ts
// SafeTensors file format parser and weight uploader for WebGPU execution.
// Operates on ArrayBuffers using native TypedArrays and DataViews. No external libraries.

import { GPUTensor } from "../gpu/tensor";

export interface TensorMeta {
    dtype: string;
    shape: number[];
    dataOffset: number; // Offset from start of raw data block (in bytes)
    dataLength: number; // Length of tensor data block (in bytes)
}

/**
 * Parses the structural JSON header of a SafeTensors binary file.
 * SafeTensors layout structure:
 * - Bytes 0-7: Little-endian u64 integer representing length of the JSON header (H).
 * - Bytes 8 to 8+H: UTF-8 encoded plain JSON string.
 * - Bytes 8+H onwards: Raw binary weight data.
 */
export function parseSafetensorsHeader(buffer: ArrayBuffer): {
    header: Map<string, TensorMeta>;
    headerLength: number;
} {
    const view = new DataView(buffer);

    // SafeTensors header length is a 64-bit unsigned integer (8 bytes)
    // We read the low and high 32-bit words separately to ensure compatibility across older environments and avoid BigInt compiler requirements
    const low = view.getUint32(0, true);
    const high = view.getUint32(4, true);
    const headerLength = low + high * 0x100000000;

    if (buffer.byteLength < 8 + headerLength) {
        throw new Error(
            `Malformed SafeTensors file: file size (${buffer.byteLength} bytes) is too small ` +
            `to accommodate the declared header length of ${headerLength} bytes.`
        );
    }

    // Parse UTF-8 JSON header slice
    const headerBytes = new Uint8Array(buffer, 8, headerLength);
    const decoder = new TextDecoder("utf-8");
    const headerStr = decoder.decode(headerBytes);

    let jsonHeader: Record<string, any>;
    try {
        jsonHeader = JSON.parse(headerStr);
    } catch (e: any) {
        throw new Error(`Failed to parse SafeTensors JSON header: ${e.message}`);
    }

    const headerMap = new Map<string, TensorMeta>();

    for (const [key, value] of Object.entries(jsonHeader)) {
        // Skip metadata key
        if (key === "__metadata__") continue;

        const dtype = value.dtype as string;
        const shape = value.shape as number[];
        const dataOffsets = value.data_offsets as [number, number];

        if (!dtype || !shape || !dataOffsets || dataOffsets.length !== 2) {
            throw new Error(`Invalid tensor header metadata format for key: ${key}`);
        }

        const dataOffset = dataOffsets[0];
        const dataLength = dataOffsets[1] - dataOffsets[0];

        headerMap.set(key, {
            dtype,
            shape,
            dataOffset,
            dataLength,
        });
    }

    return { header: headerMap, headerLength };
}

/**
 * Uploads SafeTensors array slices from the file buffer directly to the GPU as GPUTensors.
 * Prints shape details and sizing metrics to developer display console.
 */
export function loadWeightsToGPU(
    device: GPUDevice,
    buffer: ArrayBuffer,
    headerData: { header: Map<string, TensorMeta>; headerLength: number }
): Map<string, GPUTensor> {
    const { header, headerLength } = headerData;
    const weightMap = new Map<string, GPUTensor>();
    const rawDataStart = 8 + headerLength;

    console.log("%c--- Uploading SmolLM2 weights to GPU ---", "color: #8b5cf6; font-weight: bold;");

    for (const [name, meta] of header.entries()) {
        if (meta.dtype !== "F32") {
            // In early stages, we query floats only. Throw a descriptive alert if f16 shows up unexpectedly.
            throw new Error(`Weight tensor '${name}' uses non-supported format '${meta.dtype}'. Axon currently supports F32 weights.`);
        }

        const byteOffset = rawDataStart + meta.dataOffset;
        const floatElements = meta.dataLength / Float32Array.BYTES_PER_ELEMENT;

        // WebGPU Alignment Constraint: byteOffset must be key multiples of 4 for Float32Array view instantiation.
        // SafeTensors format automatically aligns all data blocks on 8-byte boundaries.
        if (byteOffset % 4 !== 0) {
            throw new Error(`Align mismatch: absolute buffer indexing offset ${byteOffset} is not a multiple of 4.`);
        }

        // Zero-copy array view projection to the underlying ArrayBuffer slice
        const floatView = new Float32Array(buffer, byteOffset, floatElements);

        // Create GPUTensor representation and upload
        const tensor = GPUTensor.fromFloat32Array(device, floatView, meta.shape);
        weightMap.set(name, tensor);

        // Calculate details and printable sizes
        const sizeInMB = meta.dataLength / (1024 * 1024);
        console.log(
            ` -> Uploaded: %c${name}%c | Shape: [${meta.shape.join(", ")}] | Size: ${sizeInMB.toFixed(2)} MB`,
            "color: #06b6d4; font-weight: bold;",
            "color: inherit;"
        );
    }

    console.log("%cAll weights uploaded successfully to GPU memory.", "color: #10b981; font-weight: bold;");
    return weightMap;
}
