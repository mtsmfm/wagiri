const dict = {
  ja: {
    'step1.title': '1. 音声ファイルを読み込む',
    'dropzone.idle': 'ここにドロップ、またはクリックして選択(wav / mp3 / m4a など)',
    'dropzone.loading': '読み込み中: {name}',
    'dropzone.loaded': '読み込み済み: {name} — 別のファイルをドロップで差し替え',
    'step2.title': '2. BGM 分離(ブラウザ内で実行)',
    'btn.separate': 'BGM とボーカルを分離',
    'sep.hint': '初回はモデルをダウンロードします(2回目以降はブラウザにキャッシュ)。音声はどこにも送信されず、すべてこのブラウザ内で処理されます。',
    'step3.title': '3. 波形編集 — 切り出し',
    'btn.play': '再生 / 停止',
    'btn.playRegion': '選択範囲を再生',
    'btn.clear': '選択解除',
    'btn.export': '選択範囲を WAV で保存',
    'edit.hint': '波形をドラッグで範囲選択。<kbd>ホイール</kbd> でズーム。範囲の端をドラッグで微調整。',
    'status.selection': '選択: {start} – {end} ({dur}s)',
    'alert.selectFirst': '先に波形をドラッグして範囲を選択してください',
    'model.fromCache': 'モデルをキャッシュから読み込み中…',
    'model.fetchFailed': 'モデルの取得に失敗 ({status})',
    'model.downloading': 'モデルをダウンロード中… {loaded}MB{total}',
    'model.init': 'モデルを初期化中…',
    'sep.progress': '分離中… {percent}% ({current}/{total})',
    'sep.nan': '⚠️ 出力異常: NaN が {count} サンプル (GPU 実行が数値破綻)。この結果は無音になります',
    'sep.zeroVocals': '⚠️ 出力異常: ボーカルが完全ゼロ (mix RMS {rms})',
    'sep.done': '完了。クリックでエディタに読み込み:',
    'sep.error': 'エラー: {message}',
    'stem.vocals': '🎤 ボーカル(声のみ)',
    'stem.bgm': '🎵 BGM のみ',
    'stem.downloadTitle': '{filename} をダウンロード',
  },
  en: {
    'step1.title': '1. Load an audio file',
    'dropzone.idle': 'Drop a file here, or click to choose (wav / mp3 / m4a, …)',
    'dropzone.loading': 'Loading: {name}',
    'dropzone.loaded': 'Loaded: {name} — drop another file to replace',
    'step2.title': '2. Separate BGM (runs in your browser)',
    'btn.separate': 'Separate BGM and vocals',
    'sep.hint': 'The model is downloaded on first use (cached by the browser afterwards). Your audio is never uploaded anywhere — everything runs inside this browser.',
    'step3.title': '3. Edit waveform — cut out a region',
    'btn.play': 'Play / Pause',
    'btn.playRegion': 'Play selection',
    'btn.clear': 'Clear selection',
    'btn.export': 'Save selection as WAV',
    'edit.hint': 'Drag on the waveform to select a range. <kbd>wheel</kbd> to zoom. Drag the edges of the range to fine-tune.',
    'status.selection': 'Selection: {start} – {end} ({dur}s)',
    'alert.selectFirst': 'Drag on the waveform to select a range first',
    'model.fromCache': 'Loading model from cache…',
    'model.fetchFailed': 'Failed to fetch model ({status})',
    'model.downloading': 'Downloading model… {loaded}MB{total}',
    'model.init': 'Initializing model…',
    'sep.progress': 'Separating… {percent}% ({current}/{total})',
    'sep.nan': '⚠️ Bad output: {count} NaN samples (GPU execution broke down numerically). This result will be silent',
    'sep.zeroVocals': '⚠️ Bad output: vocals are all zero (mix RMS {rms})',
    'sep.done': 'Done. Click to load into the editor:',
    'sep.error': 'Error: {message}',
    'stem.vocals': '🎤 Vocals only',
    'stem.bgm': '🎵 BGM only',
    'stem.downloadTitle': 'Download {filename}',
  },
}

const LANG_KEY = 'wagiri-lang'
const stored = localStorage.getItem(LANG_KEY)
export const lang = stored === 'ja' || stored === 'en'
  ? stored
  : (navigator.language || 'en').toLowerCase().startsWith('ja') ? 'ja' : 'en'

export function t(key, params = {}) {
  const s = dict[lang][key] ?? dict.en[key] ?? key
  return s.replace(/\{(\w+)\}/g, (_, k) => (params[k] !== undefined ? params[k] : `{${k}}`))
}

export function applyI18n() {
  document.documentElement.lang = lang
  for (const el of document.querySelectorAll('[data-i18n]')) el.textContent = t(el.dataset.i18n)
  for (const el of document.querySelectorAll('[data-i18n-html]')) el.innerHTML = t(el.dataset.i18nHtml)
  const sel = document.getElementById('lang-select')
  if (sel) {
    sel.value = lang
    // Reload on switch: dynamic strings resolve via t() at call time, so the
    // simplest way to make everything consistent is a fresh page load
    sel.addEventListener('change', () => {
      localStorage.setItem(LANG_KEY, sel.value)
      location.reload()
    })
  }
}
