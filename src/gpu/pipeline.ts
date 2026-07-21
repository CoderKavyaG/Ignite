/**
 * Utility to compile a WGSL shader module and create a GPUComputePipeline.
 * Enables rapid setup of WebGPU compute kernels.
 *
 * WebGPU Pipeline Layout:
 * - We use layout: "auto". This informs the WebGPU compiler to inspect the shader module, 
 *   compute the necessary BindGroup layouts from the bindings defined in the WGSL,
 *   and resolve WebGPU pipeline bindings without requiring manual layouts.
 *
 * @param device The WebGPU GPUDevice context.
 * @param wgslCode The raw string containing WGSL shader code.
 * @param entryPoint The name of the entry function marked with @compute in the WGSL. Defaults to "main".
 */
export function createComputePipeline(
    device: GPUDevice,
    wgslCode: string,
    entryPoint: string = "main"
): GPUComputePipeline {
    const shaderModule = device.createShaderModule({
        code: wgslCode,
    });

    // Create and return the compute pipeline.
    // Note: If WGSL compilation fails, device.createComputePipeline will trigger a WebGPU validation error,
    // which outputs detailed compiler logs in the developer console.
    return device.createComputePipeline({
        layout: "auto",
        compute: {
            module: shaderModule,
            entryPoint: entryPoint,
        },
    });
}
