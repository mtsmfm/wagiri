const dict = {
  ja: {
    'step1.title': '1. 音声ファイルを読み込む',
    'dropzone.idle': '音声ファイルをドロップ、または選択',
    'dropzone.loading': '読み込み中: {name}',
    'dropzone.loaded': '読み込み済み: {name}',
    'step2.title': '2. BGM 分離',
    'btn.separate': '分離する',
    'sep.hint': '初回のみモデルをダウンロードします。音声は外部に送信されません。',
    'step3.title': '3. 切り出し',
    'btn.play': '再生 / 停止',
    'btn.playRegion': '選択を再生',
    'btn.clear': '選択解除',
    'btn.export': '選択を WAV で保存',
    'edit.hint': 'ドラッグで範囲選択。<kbd>ホイール</kbd> でズーム。',
    'status.selection': '選択: {start} – {end} ({dur}s)',
    'alert.selectFirst': '範囲を選択してください',
    'model.fromCache': 'モデルをキャッシュから読み込み中…',
    'model.fetchFailed': 'モデルの取得に失敗 ({status})',
    'model.downloading': 'モデルをダウンロード中… {loaded}MB{total}',
    'model.init': 'モデルを初期化中…',
    'sep.progress': '分離中… {percent}% ({current}/{total})',
    'sep.nan': '⚠️ 分離に失敗しました (NaN: {count})',
    'sep.zeroVocals': '⚠️ 分離に失敗しました (ボーカルが無音、mix RMS: {rms})',
    'sep.done': '完了',
    'sep.error': 'エラー: {message}',
    'stem.vocals': '🎤 ボーカル',
    'stem.bgm': '🎵 BGM のみ',
    'stem.downloadTitle': '{filename} をダウンロード',
  },
  en: {
    'step1.title': '1. Load an audio file',
    'dropzone.idle': 'Drop or choose an audio file',
    'dropzone.loading': 'Loading: {name}',
    'dropzone.loaded': 'Loaded: {name}',
    'step2.title': '2. Separate BGM',
    'btn.separate': 'Separate',
    'sep.hint': 'The model is downloaded once. Your audio is never uploaded.',
    'step3.title': '3. Trim',
    'btn.play': 'Play / Pause',
    'btn.playRegion': 'Play selection',
    'btn.clear': 'Clear selection',
    'btn.export': 'Save selection as WAV',
    'edit.hint': 'Drag to select. <kbd>wheel</kbd> to zoom.',
    'status.selection': 'Selection: {start} – {end} ({dur}s)',
    'alert.selectFirst': 'Select a range first',
    'model.fromCache': 'Loading model from cache…',
    'model.fetchFailed': 'Failed to fetch model ({status})',
    'model.downloading': 'Downloading model… {loaded}MB{total}',
    'model.init': 'Initializing model…',
    'sep.progress': 'Separating… {percent}% ({current}/{total})',
    'sep.nan': '⚠️ Separation failed (NaN: {count})',
    'sep.zeroVocals': '⚠️ Separation failed (silent vocals, mix RMS: {rms})',
    'sep.done': 'Done',
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
