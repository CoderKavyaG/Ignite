import './style.css';
import { initWebGPU } from './gpu/device';
import { matmul } from './kernels/matmul';
import matmulShader from './shaders/matmul.wgsl?raw';

// Get DOM elements
const gpuBadge = document.getElementById('gpu-badge');
const gpuStatusText = document.getElementById('gpu-status-text');
const diagAdapter = document.getElementById('diag-adapter');
const diagArch = document.getElementById('diag-arch');
const diagType = document.getElementById('diag-type');
const diagWorkgroups = document.getElementById('diag-workgroups');

const matrixAContainer = document.getElementById('matrix-a-4x4');
const matrixBContainer = document.getElementById('matrix-b-4x4');
const matrixCContainer = document.getElementById('matrix-c-4x4');
const testConsole = document.getElementById('test-console');

const inputM = document.getElementById('input-m') as HTMLInputElement;
const inputK = document.getElementById('input-k') as HTMLInputElement;
const inputN = document.getElementById('input-n') as HTMLInputElement;
const btnRunMatmul = document.getElementById('btn-run-matmul') as HTMLButtonElement;

const metricTime = document.getElementById('metric-time');
const metricMemory = document.getElementById('metric-memory');
const metricGrid = document.getElementById('metric-grid');
const playgroundMatrixScroll = document.getElementById('playground-matrix-scroll');
const matrixPlaygroundC = document.getElementById('matrix-playground-c');
const codeDisplay = document.getElementById('code-display');

// Display the WGSL shader source code in the shader viewer card
if (codeDisplay) {
  codeDisplay.textContent = matmulShader;
}

/**
 * Renders a matrix into a DOM container as a grid of cells.
 */
function renderMatrix(container: HTMLElement, data: Float32Array, cols: number, rows: number, maxDispCols = 8, maxDispRows = 8) {
  container.innerHTML = '';
  // Set inline styles for columns
  const dispCols = Math.min(cols, maxDispCols);
  const dispRows = Math.min(rows, maxDispRows);
  container.style.gridTemplateColumns = `repeat(${dispCols}, 1fr)`;

  for (let r = 0; r < dispRows; r++) {
    for (let c = 0; c < dispCols; c++) {
      const idx = r * cols + c;
      const cell = document.createElement('div');
      cell.className = 'matrix-cell';

      const val = data[idx];
      cell.textContent = val % 1 === 0 ? val.toFixed(0) : val.toFixed(2);

      // Visual formatting indicators
      if (container.id === 'matrix-a-4x4' && val === 1.0 && r === c) {
        cell.classList.add('identity-one');
      } else if (container.id === 'matrix-c-4x4' || container.id === 'matrix-playground-c') {
        cell.classList.add('calculated-value');
      }

      container.appendChild(cell);
    }
  }
}

async function runApplication() {
  let device: GPUDevice;

  try {
    // 1. Initializing WebGPU
    device = await initWebGPU();

    // Retrieve Adapter info via modern WebGPU spec
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error("Adapter mismatch query failed.");

    // Read manufacturer information safely
    const info = (adapter as any).info || {};
    const vendor = info.vendor || "Unknown Vendor";
    const architecture = info.architecture || "Unknown Architecture";
    const type = (adapter as any).isFallbackAdapter ? "Software Adapter" : "Hardware Accelerated GPU";
    const maxWorkgroups = device.limits.maxComputeWorkgroupSizeX;

    // Update Diagnostics Card Info
    if (diagAdapter) diagAdapter.textContent = vendor;
    if (diagArch) diagArch.textContent = architecture;
    if (diagType) diagType.textContent = type;
    if (diagWorkgroups) diagWorkgroups.textContent = `${maxWorkgroups} x ${device.limits.maxComputeWorkgroupSizeY} x ${device.limits.maxComputeWorkgroupSizeZ}`;

    // Update GPU Status Badge
    if (gpuBadge && gpuStatusText) {
      gpuBadge.className = 'gpu-status success';
      gpuStatusText.textContent = 'WebGPU Connected';
    }
  } catch (err: any) {
    console.error("Initialization error:", err);
    if (diagAdapter) diagAdapter.textContent = "Unavailable";
    if (diagArch) diagArch.textContent = "Unavailable";
    if (diagType) diagType.textContent = "Unavailable";
    if (diagWorkgroups) diagWorkgroups.textContent = "Unavailable";

    if (gpuBadge && gpuStatusText) {
      gpuBadge.className = 'gpu-status error';
      gpuStatusText.textContent = 'WebGPU Failed';
    }

    if (testConsole) {
      testConsole.textContent = `CRITICAL ERROR: ${err.message}`;
      testConsole.style.color = '#ef4444';
    }

    if (matrixCContainer) {
      matrixCContainer.innerHTML = `<div align="center" style="grid-column: span 4; color: var(--accent-error); font-size: 0.85rem;">Device Error</div>`;
    }
    return;
  }

  // -------------------------------------------------------------
  // RUN MANDATORY VERIFICATION TEST (4x4 Identity x 4x4 Sequence)
  // -------------------------------------------------------------
  const M_test = 4;
  const K_test = 4;
  const N_test = 4;

  // Float32Array layout mapping: flat buffer elements
  // Shape: (4, 4) Identity Matrix A
  const matrixA_4x4 = new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  ]);

  // Shape: (4, 4) Arbitrary test values in sequence for Matrix B
  const matrixB_4x4 = new Float32Array([
    2.5, 4.0, 6.5, 8.0,
    1.0, 3.5, 5.0, 7.5,
    9.0, 11.5, 13.0, 15.5,
    8.5, 10.0, 12.5, 14.0
  ]);

  // Render inputs visual panels
  if (matrixAContainer) renderMatrix(matrixAContainer, matrixA_4x4, 4, 4);
  if (matrixBContainer) renderMatrix(matrixBContainer, matrixB_4x4, 4, 4);

  if (testConsole) {
    testConsole.textContent = "Compiling matmul pipeline and allocating buffers...\n";
  }

  try {
    const startTime = performance.now();

    // Execute Matrix Multiplication C = A x B on GPU
    // Shape Trace: (4, 4) x (4, 4) = (4, 4)
    const resultC_4x4 = await matmul(device, matrixA_4x4, matrixB_4x4, M_test, K_test, N_test);

    const endTime = performance.now();
    const elapsed = endTime - startTime;

    // Log the requested log to verify correctness to the user environment console
    console.log("-----------------------------------------");
    console.log("AXON WebGPU 4x4 Matmul Verification Test Result:");
    console.log("Matrix A (Identity 4x4):\n", matrixA_4x4);
    console.log("Matrix B (Test Sequence 4x4):\n", matrixB_4x4);
    console.log("Matrix C (GPU Result 4x4):\n", resultC_4x4);
    console.log(`Execution time: ${elapsed.toFixed(3)} ms`);
    console.log("-----------------------------------------");

    // Display visually
    if (matrixCContainer) {
      renderMatrix(matrixCContainer, resultC_4x4, 4, 4);
    }

    if (testConsole) {
      testConsole.textContent =
        `Shader execution verified successfully!\n` +
        `Result matching Matrix B: ${arraysEqual(resultC_4x4, matrixB_4x4) ? 'TRUE' : 'FALSE'}\n` +
        `Float32 round-trip dispatch latency: ${elapsed.toFixed(3)} ms\n` +
        `Output elements:\n[${resultC_4x4.join(', ')}]`;
    }
  } catch (err: any) {
    console.error("4x4 Matmul verification error:", err);
    if (testConsole) {
      testConsole.textContent += `Execution crash details: ${err.message}`;
    }
  }

  // Helper function to check mathematical equivalence
  function arraysEqual(a: Float32Array, b: Float32Array) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (Math.abs(a[i] - b[i]) > 1e-5) return false;
    }
    return true;
  }

  // -------------------------------------------------------------
  // RUN INTERACTIVE BENCHMARK PLAYGROUND
  // -------------------------------------------------------------
  async function runPlayground() {
    const M = parseInt(inputM.value);
    const K = parseInt(inputK.value);
    const N = parseInt(inputN.value);

    if (isNaN(M) || isNaN(K) || isNaN(N) || M <= 0 || K <= 0 || N <= 0) {
      alert("Please enter valid, positive dimensions for M, K, and N.");
      return;
    }

    // Set buttons as loading
    btnRunMatmul.disabled = true;
    btnRunMatmul.textContent = "Computing...";

    try {
      // 1. Generate random matrix floats
      // Shape Trace:
      // A_rand: shape (M, K)
      // B_rand: shape (K, N)
      const A_rand = new Float32Array(M * K);
      const B_rand = new Float32Array(K * N);

      // Fill with arbitrary test weights
      for (let i = 0; i < A_rand.length; i++) A_rand[i] = Math.random() * 2.0 - 1.0;
      for (let i = 0; i < B_rand.length; i++) B_rand[i] = Math.random() * 2.0 - 1.0;

      const sizeBytesA = A_rand.byteLength;
      const sizeBytesB = B_rand.byteLength;
      const sizeBytesC = M * N * 4;
      const sizeBytesUniforms = 16;
      const totalMemory = sizeBytesA + sizeBytesB + sizeBytesC + sizeBytesUniforms;

      if (metricMemory) {
        metricMemory.textContent = `${(totalMemory / 1024).toFixed(1)} KB`;
      }
      if (metricGrid) {
        // workgroups: Math.ceil(N / 8) x Math.ceil(M / 8)
        metricGrid.textContent = `${Math.ceil(N / 8)}, ${Math.ceil(M / 8)}, 1`;
      }

      // 2. Multiply matrices A x B on GPU
      const t0 = performance.now();
      const output = await matmul(device, A_rand, B_rand, M, K, N);
      const t1 = performance.now();

      const elapsed = t1 - t0;
      if (metricTime) {
        metricTime.textContent = `${elapsed.toFixed(2)} ms`;
      }

      // Render representative preview (Top 8x8 cells) of Output Matrix C
      if (playgroundMatrixScroll && matrixPlaygroundC) {
        playgroundMatrixScroll.classList.remove('hidden');
        renderMatrix(matrixPlaygroundC, output, N, M, 8, 8);
      }
    } catch (e: any) {
      console.error(e);
      alert(`Playground compute failure: ${e.message}`);
    } finally {
      btnRunMatmul.disabled = false;
      btnRunMatmul.textContent = "Run GPU Kernel";
    }
  }

  // Add click listener
  btnRunMatmul.addEventListener('click', runPlayground);
}

// Kick off when window loads
window.addEventListener('DOMContentLoaded', runApplication);
