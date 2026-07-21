import { defineConfig } from 'vite';

// Cross-Origin-Opener-Policy (COOP) and Cross-Origin-Embedder-Policy (COEP) headers
// are required to establish cross-origin isolation for the browser tab.
// This is essential when building high-performance applications (like our LLM engine) because:
// 1. It enables the use of `SharedArrayBuffer`, which allows zero-copy, efficient memory sharing
//    between the main thread and Web Workers (useful for pipelining token generation and GPU commands).
// 2. It enables high-resolution timers (`performance.now()`), which are otherwise restricted/coarsened
//    due to Spectre/Meltdown security mitigations. This is critical for micro-benchmarking our WebGPU kernels.
export default defineConfig({
    server: {
        headers: {
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Embedder-Policy': 'require-corp',
        },
    },
});
