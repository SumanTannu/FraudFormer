import { useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  FileText,
  LoaderCircle,
  RefreshCw,
  ScanSearch,
  ShieldCheck,
  Upload,
  X,
} from 'lucide-react'

const SAMPLE = `Customer Support Specialist

We are looking for a customer-focused team member to answer product questions and support account holders. The successful candidate will receive full training and work remotely.

Before an interview can be arranged, applicants must send a copy of their passport, bank account details, and online banking password to the recruitment manager. Immediate availability is required.`

const RESULT_META = {
  'Legitimate': {
    tone: 'safe',
    title: 'Likely legitimate',
    description: 'The language patterns are most consistent with a legitimate job advertisement.',
    Icon: CheckCircle2,
  },
  'Human-written Fraud': {
    tone: 'danger',
    title: 'Potential human-written fraud',
    description: 'The posting contains patterns associated with fraudulent advertisements written by people.',
    Icon: AlertTriangle,
  },
  'AI-generated Fraud': {
    tone: 'warning',
    title: 'Potential AI-generated fraud',
    description: 'The posting contains patterns associated with AI-generated fraudulent advertisements.',
    Icon: ScanSearch,
  },
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`
}

function App() {
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('fraudformer-deberta')
  const [text, setText] = useState('')
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [apiState, setApiState] = useState('checking')
  const inputRef = useRef(null)

  useEffect(() => {
    Promise.all([
      fetch('/api/health').then((response) => {
        if (!response.ok) throw new Error('Prediction API is unavailable')
        return response.json()
      }),
      fetch('/api/models').then((response) => {
        if (!response.ok) throw new Error('Could not load models')
        return response.json()
      }),
    ])
      .then(([health, modelData]) => {
        setApiState(health.device === 'cuda' ? 'GPU ready' : 'CPU ready')
        setModels(modelData.models)
        const recommended = modelData.models.find((model) => model.recommended)
        if (recommended) setSelectedModel(recommended.id)
      })
      .catch((problem) => {
        setApiState('API offline')
        setError(`${problem.message}. Start the Python API on port 8000.`)
      })
  }, [])

  const activeModel = useMemo(
    () => models.find((model) => model.id === selectedModel),
    [models, selectedModel],
  )

  async function readFile(uploaded) {
    if (!uploaded) return
    const extension = uploaded.name.split('.').pop()?.toLowerCase()
    if (!['txt', 'md', 'csv', 'json'].includes(extension)) {
      setError('Please upload a TXT, MD, CSV, or JSON text file.')
      return
    }
    if (uploaded.size > 1_000_000) {
      setError('The file must be smaller than 1 MB.')
      return
    }
    try {
      const contents = await uploaded.text()
      if (!contents.trim()) throw new Error('The selected file is empty.')
      setFile(uploaded)
      setText(contents)
      setResult(null)
      setError('')
    } catch (problem) {
      setError(problem.message)
    }
  }

  function clearFile() {
    setFile(null)
    setText('')
    setResult(null)
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  async function analyze() {
    if (text.trim().length < 20) {
      setError('Add a complete job posting with at least 20 characters.')
      return
    }
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel, text }),
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || 'Prediction failed')
      setResult(payload)
    } catch (problem) {
      setError(problem.message)
    } finally {
      setLoading(false)
    }
  }

  const meta = result ? RESULT_META[result.label_name] : null

  return (
    <main className="page-shell">
      <header className="topbar">
        <a className="brand" href="#classifier" aria-label="FraudFormer classifier">
          <span className="brand-mark"><ShieldCheck size={21} strokeWidth={2.2} /></span>
          <span>FraudFormer</span>
        </a>
        <div className={`status-pill ${apiState === 'API offline' ? 'offline' : ''}`}>
          <span className="status-dot" /> {apiState}
        </div>
      </header>

      <section className="workspace" id="classifier" aria-label="Job posting classifier">
        <div className="input-panel">
          <div className="section-heading">
            <div>
              <span className="step">01</span>
              <h2>Add a job posting</h2>
            </div>
            {text && <span className="character-count">{text.length.toLocaleString()} characters</span>}
          </div>

          {!file ? (
            <button className="drop-zone" type="button" onClick={() => inputRef.current?.click()}>
              <span className="upload-icon"><Upload size={23} /></span>
              <strong>Upload a job posting</strong>
              <span>TXT, MD, CSV, or JSON · up to 1 MB</span>
            </button>
          ) : (
            <div className="file-chip">
              <span className="file-icon"><FileText size={20} /></span>
              <span><strong>{file.name}</strong><small>{(file.size / 1024).toFixed(1)} KB</small></span>
              <button type="button" onClick={clearFile} aria-label="Remove uploaded file"><X size={18} /></button>
            </div>
          )}
          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept=".txt,.md,.csv,.json,text/plain,text/markdown,application/json,text/csv"
            onChange={(event) => readFile(event.target.files?.[0])}
          />

          <div className="divider"><span>or paste the text</span></div>
          <textarea
            value={text}
            onChange={(event) => {
              setText(event.target.value)
              setResult(null)
            }}
            placeholder="Paste the full title, company profile, description, requirements, and benefits here…"
            aria-label="Job posting text"
          />
          <button className="sample-button" type="button" onClick={() => { setText(SAMPLE); setFile(null); setResult(null) }}>
            Use a suspicious example
          </button>
        </div>

        <aside className="control-panel">
          <div className="section-heading">
            <div>
              <span className="step">02</span>
              <h2>Choose a model</h2>
            </div>
          </div>
          <label className="select-wrap">
            <span className="visually-hidden">Prediction model</span>
            <select value={selectedModel} onChange={(event) => { setSelectedModel(event.target.value); setResult(null) }} disabled={!models.length}>
              {models.length ? models.map((model) => (
                <option key={model.id} value={model.id}>{model.name}</option>
              )) : <option>Loading models…</option>}
            </select>
            <ChevronDown size={18} aria-hidden="true" />
          </label>
          <div className="model-note">
            {activeModel?.recommended && <span className="recommended">Recommended</span>}
            <strong>{activeModel?.name || 'Connecting to model registry'}</strong>
            <p>{activeModel?.description || 'Available checkpoints will appear here.'}</p>
          </div>
          <button className="analyze-button" type="button" onClick={analyze} disabled={loading || !models.length}>
            {loading ? <><LoaderCircle className="spinner" size={20} /> Loading model & analysing…</> : <><ScanSearch size={20} /> Analyse posting</>}
          </button>
          <p className="privacy-note">Your text is sent only to the local prediction server.</p>
        </aside>
      </section>

      {error && (
        <div className="error-banner" role="alert">
          <AlertTriangle size={19} /> <span>{error}</span>
          <button type="button" onClick={() => setError('')} aria-label="Dismiss error"><X size={17} /></button>
        </div>
      )}

      {result && meta && (
        <section className={`result-card ${meta.tone}`} aria-live="polite">
          <div className="result-summary">
            <span className="result-icon"><meta.Icon size={29} /></span>
            <div>
              <p className="eyebrow">Classification result</p>
              <h2>{meta.title}</h2>
              <p>{meta.description}</p>
            </div>
            <div className="confidence">
              <strong>{formatPercent(result.confidence)}</strong>
              <span>confidence</span>
            </div>
          </div>
          <div className="probability-grid">
            {result.probabilities.map((probability) => (
              <div className="probability" key={probability.label}>
                <div><span>{probability.label}</span><strong>{formatPercent(probability.value)}</strong></div>
                <div className="bar"><span style={{ width: formatPercent(probability.value) }} /></div>
              </div>
            ))}
          </div>
          <footer className="result-footer">
            <span>Model: <strong>{result.model_name}</strong></span>
            <span>Inference: <strong>{result.latency_ms.toLocaleString()} ms</strong></span>
            <button type="button" onClick={() => { setResult(null); window.scrollTo({ top: 0, behavior: 'smooth' }) }}>
              <RefreshCw size={15} /> Analyse another
            </button>
          </footer>
        </section>
      )}

    </main>
  )
}

export default App
