export interface GPUContextBundle {
  adapter: GPUAdapter;
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
}

/** Shared WebGPU setup. Throws a readable Error if WebGPU is unavailable. */
export async function initWebGPU(canvas: HTMLCanvasElement): Promise<GPUContextBundle> {
  if (!('gpu' in navigator) || !navigator.gpu) {
    throw new Error('WebGPU is not available in this browser. Use a recent Chrome or Edge.');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter found (GPU blocklisted or disabled?).');

  // Ask for the adapter's full buffer limits: 100k particles * 144 B is ~14 MB,
  // and big grids need more than the 128 MB default binding size later on.
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  device.lost.then((info) => console.error(`WebGPU device lost (${info.reason}): ${info.message}`));
  device.addEventListener('uncapturederror', (e) =>
    console.error('WebGPU error:', (e as GPUUncapturedErrorEvent).error.message),
  );

  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('Could not get a WebGPU canvas context.');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  return { adapter, device, context, format };
}

/** Keeps the canvas backing store matched to its CSS size. Returns true if it changed. */
export function resizeCanvasToDisplaySize(canvas: HTMLCanvasElement, maxDim = 4096): boolean {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.min(maxDim, Math.round(canvas.clientWidth * dpr)));
  const h = Math.max(1, Math.min(maxDim, Math.round(canvas.clientHeight * dpr)));
  if (canvas.width === w && canvas.height === h) return false;
  canvas.width = w;
  canvas.height = h;
  return true;
}

export function describeAdapter(adapter: GPUAdapter): string {
  const i = adapter.info;
  const parts = [i?.vendor, i?.architecture, i?.device, i?.description].filter(Boolean);
  return parts.length ? parts.join(' / ') : 'unknown adapter';
}
