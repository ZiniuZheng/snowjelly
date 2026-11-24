/**
 * Saves a canvas as PNG. In `npm run dev` the file is written straight to
 * screenshots/<name>.png in the repo (see vite.config.ts); in a built page
 * it falls back to a normal browser download.
 *
 * For a WebGPU canvas, call this right after submitting a frame (or create
 * the context with preserveDrawingBuffer-like behavior) so it is not blank.
 */
export async function saveCanvasPNG(canvas: HTMLCanvasElement, name: string): Promise<string> {
  const blob = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/png'),
  );
  if (import.meta.env.DEV) {
    const res = await fetch(`/__screenshot?name=${encodeURIComponent(name)}`, { method: 'POST', body: blob });
    const { file } = (await res.json()) as { file: string };
    console.log(`Saved ${file}`);
    return file;
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${name}.png`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  return a.download;
}
