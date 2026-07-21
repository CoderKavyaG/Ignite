let device: GPUDevice | null = null;

/**
 * Initializes the WebGPU context by checking support, requesting a high-performance adapter,
 * and creating the GPUDevice singleton.
 * Throws descriptive errors if WebGPU is unavailable or fails to initialize.
 */
export async function initWebGPU(): Promise<GPUDevice> {
    if (device) return device;

    if (!navigator.gpu) {
        throw new Error(
            "WebGPU is not supported by your browser or environment. " +
            "Please use a modern browser (such as Chrome 113+, Edge 113+, Opera, or Safari 17+) " +
            "and ensure WebGPU flags/features are enabled on your hardware."
        );
    }

    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance", // Request the discrete GPU if available for maximum throughput
    });

    if (!adapter) {
        throw new Error(
            "WebGPU support detected, but no suitable GPU adapter was found. " +
            "This can happen if graphics drivers are outdated, WebGPU is blacklisted for your GPU, " +
            "or hardware acceleration is disabled in your browser settings."
        );
    }

    device = await adapter.requestDevice();

    // Set up device loss handler. Loss can happen due to driver crashes, physical GPU removal,
    // or browser resets. Catching this is a production-level requirement.
    device.lost.then((info) => {
        console.warn(`WebGPU device was lost: ${info.message} (Reason: ${info.reason})`);
        device = null; // Mark for re-initialization if needed
    });

    return device;
}

/**
 * Returns the initialized GPUDevice singleton.
 * Throws an error if initWebGPU has not been called and completed yet.
 */
export function getDevice(): GPUDevice {
    if (!device) {
        throw new Error("WebGPU GPUDevice singleton requested before initialization. Call initWebGPU() first.");
    }
    return device;
}
