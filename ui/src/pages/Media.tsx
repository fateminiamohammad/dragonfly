import { useState } from 'react';
import { api } from '../api';
import { pct } from '../format';

interface OcrLine {
  text: string;
  confidence: number;
}
interface Tag {
  label: string;
  score: number;
}
interface PerceiveItem {
  type: 'image' | 'audio';
  ocr?: { text: string; lines: OcrLine[] };
  tags?: Tag[];
  transcript?: string;
  language?: string;
  duration_s?: number;
}
interface PerceiveResponse {
  items: Record<string, PerceiveItem>;
  latency_ms: Record<string, number>;
}

/** File -> base64 (without the data: prefix). */
export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',', 2)[1] ?? '');
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

export function mediaType(file: File): 'image' | 'audio' | null {
  if (file.type.startsWith('image/')) return 'image';
  if (file.type.startsWith('audio/') || file.type === 'video/webm' || file.type === 'video/ogg') return 'audio';
  return null;
}

export function Media() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [language, setLanguage] = useState('en');
  const [ocrLang, setOcrLang] = useState('en');
  const [result, setResult] = useState<PerceiveResponse | null>(null);
  const [roundTrip, setRoundTrip] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const kind = file ? mediaType(file) : null;

  function pick(f: File | null) {
    setResult(null);
    setError('');
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setPreview(f ? URL.createObjectURL(f) : '');
    if (f && !mediaType(f)) setError('Choose an image (png, jpg, webp…) or an audio file (wav, mp3, ogg, m4a…).');
  }

  async function run() {
    if (!file || !kind) return;
    setBusy(true);
    setError('');
    const t = performance.now();
    try {
      const item = { id: 'file', type: kind, data: await fileToBase64(file), ...(kind === 'audio' ? { language } : { ocr_lang: ocrLang }) };
      setResult(await api<PerceiveResponse>('/model/perceive', { method: 'POST', body: JSON.stringify({ items: [item] }) }));
      setRoundTrip(performance.now() - t);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const out = result?.items.file;
  return (
    <div className="split">
      <section className="card">
        <h2>Image or audio</h2>
        <p className="muted small">
          Converted to text with non-autoregressive models: OCR and tags for images, speech recognition for audio. Include
          media in a decision request and the model reads this text.
        </p>
        <input type="file" accept="image/*,audio/*" onChange={(e) => pick(e.target.files?.[0] ?? null)} aria-label="choose a file" />
        {kind === 'image' && (
          <>
            <img className="preview" src={preview} alt="selected" />
            <label className="row">
              Text language
              <select value={ocrLang} onChange={(e) => setOcrLang(e.target.value)}>
                <option value="en">English</option>
                <option value="arabic">Persian / Arabic</option>
              </select>
            </label>
          </>
        )}
        {kind === 'audio' && (
          <>
            <audio className="preview" controls src={preview} />
            <label className="row">
              Speech language
              <select value={language} onChange={(e) => setLanguage(e.target.value)}>
                <option value="en">English</option>
                <option value="fa">Persian</option>
              </select>
            </label>
          </>
        )}
        <div className="row">
          <button className="primary" onClick={run} disabled={!kind || busy}>
            {busy ? 'Converting…' : 'Convert to text'}
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      <section className="card">
        <div className="row">
          <h2>Text</h2>
          {result && (
            <span className="muted small">
              {Object.entries(result.latency_ms)
                .filter(([k]) => k !== 'total')
                .map(([k, v]) => `${k} ${v.toFixed(0)} ms`)
                .join(' · ')}{' '}
              · round trip {roundTrip?.toFixed(0)} ms
            </span>
          )}
        </div>
        {!out && <p className="muted">Choose a file and convert it.</p>}
        {out?.transcript !== undefined && (
          <>
            <h3>Transcript</h3>
            <p className="transcript">{out.transcript || <span className="muted">(no speech found)</span>}</p>
            <p className="muted small">
              {out.duration_s} s of audio · language {out.language}
            </p>
          </>
        )}
        {out?.ocr && (
          <>
            <h3>Text in the image</h3>
            {out.ocr.text ? <pre className="ocr">{out.ocr.text}</pre> : <p className="muted">(no text found)</p>}
          </>
        )}
        {out?.tags && (
          <>
            <h3>What it shows</h3>
            {out.tags.map((t) => (
              <div key={t.label} className="bar-row" title={`${t.label}: ${pct(t.score)} of the best match`}>
                <span className="bar-label">{t.label}</span>
                <span className="bar">
                  <span style={{ width: `${Math.max(t.score * 100, 0.5)}%` }} />
                </span>
                <span className="bar-value">{pct(t.score)}</span>
              </div>
            ))}
            <p className="muted small">Scores are relative (they rank tags against each other), not probabilities.</p>
          </>
        )}
      </section>
    </div>
  );
}
