import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuditStore } from '../hooks/useAuditStore'
import { runCartography, runConstitution, runProxyHunter } from '../utils/api'

// ── Model type config ──────────────────────────────────────────────────────
const MODEL_TYPES = [
  { id: 'sklearn',     icon: '🧠', label: 'scikit-learn / XGBoost',      desc: 'Upload a .pkl model file' },
  { id: 'api',         icon: '🌐', label: 'REST API Endpoint',            desc: 'Any model behind an HTTP URL' },
  { id: 'huggingface', icon: '🤗', label: 'HuggingFace',                  desc: 'Any classifier from HF Hub' },
  { id: 'openai',      icon: '🔮', label: 'OpenAI (ChatGPT / GPT-4)',     desc: 'gpt-4o, gpt-4, gpt-3.5-turbo' },
  { id: 'gemini_llm',  icon: '✦',  label: 'Gemini LLM',                   desc: 'gemini-2.0-flash, gemini-1.5-pro' },
  { id: 'vertex_ai',   icon: '☁',  label: 'Vertex AI Endpoint',           desc: 'Google Cloud deployed model' },
] as const

type ModelType = typeof MODEL_TYPES[number]['id']

// ── Test suite descriptions shown to the user ──────────────────────────────
const SUITE_INFO: Record<string, { label: string; detail: string }> = {
  text_probes: {
    label: 'Demographic Text Probes',
    detail: '260 counterfactual texts across 13 groups (race, gender, religion). Auto-selected for text/LLM models.',
  },
  hiring_bias: {
    label: 'Hiring & Employment Bias',
    detail: '500 synthetic hiring records with injected gender (+30 pp) and race (+20 pp) disparities. Auto-selected for tabular models.',
  },
}

function DropZone({ file, onDrop }: { file: File | null; onDrop: (f: File[]) => void }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'application/octet-stream': ['.pkl'] }, maxFiles: 1
  })
  return (
    <div {...getRootProps()} className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all
      ${isDragActive ? 'border-lens bg-lens/5' : file ? 'border-signal-green/40 bg-signal-green/5' : 'border-white/10 hover:border-white/20'}`}>
      <input {...getInputProps()} />
      <div className="text-2xl mb-1">{file ? '✓' : '📎'}</div>
      {file
        ? <div className="text-signal-green font-mono text-xs">{file.name}</div>
        : <div className="text-white/40 text-xs">{isDragActive ? 'Drop here' : 'Drag .pkl or click'}</div>}
    </div>
  )
}

export default function AuditPage() {
  const nav = useNavigate()
  const store = useAuditStore()
  const [modelType, setModelType] = useState<ModelType>('sklearn')
  const [apiEndpoint, setApiEndpoint] = useState('')
  const [hfModel, setHfModel] = useState('')
  const [hfToken, setHfToken] = useState('')
  const [openaiModel, setOpenaiModel] = useState('gpt-4o')
  const [openaiKey, setOpenaiKey] = useState('')
  const [geminiModel, setGeminiModel] = useState('gemini-2.0-flash')
  const [geminiKey, setGeminiKey] = useState('')
  const [vertexEndpoint, setVertexEndpoint] = useState('')
  const [gcpProject, setGcpProject] = useState('')
  const [runningStage, setRunningStage] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)

  const onDropModel = useCallback((files: File[]) => { if (files[0]) store.setModelFile(files[0]) }, [])

  const modelReady =
    modelType === 'sklearn'     ? !!store.modelFile :
    modelType === 'huggingface' ? !!hfModel :
    modelType === 'openai'      ? !!(openaiModel && openaiKey) :
    modelType === 'gemini_llm'  ? !!(geminiModel && geminiKey) :
    modelType === 'api'         ? !!apiEndpoint :
    modelType === 'vertex_ai'   ? !!(vertexEndpoint && gcpProject) :
    false

  const canRun = modelReady

  // Which built-in suite will auto-select for this model type
  const autoSuite = ['huggingface','openai','gemini_llm'].includes(modelType) ? 'text_probes' : 'hiring_bias'

  const runAudit = async () => {
    if (!canRun) return
    store.setError(null)
    store.setLoading(true)

    try {
      setRunningStage('Bias Cartography'); setProgress(10)
      store.setStage('cartography')

      const modelEndpoint =
        modelType === 'huggingface' ? hfModel :
        modelType === 'openai'      ? openaiModel :
        modelType === 'gemini_llm'  ? geminiModel :
        apiEndpoint

      const resolvedLlmKey =
        modelType === 'openai'    ? openaiKey :
        modelType === 'gemini_llm'? geminiKey : ''

      const extraParams = {
        model_type: modelType,
        api_endpoint: modelEndpoint,
        vertex_endpoint_id: vertexEndpoint,
        gcp_project: gcpProject,
        llm_api_key: resolvedLlmKey,
        hf_token: hfToken,
        test_suite: 'auto',
      }

      const carto = await runCartography(
        store.modelFile,
        ['auto'],
        'auto',
        extraParams,
      )
      store.setCartographyResults(carto)

      store.setModelType(modelType)
      store.setModelEndpoint(modelEndpoint)
      store.setLlmApiKey(resolvedLlmKey)
      store.setHfToken(hfToken)
      store.setTestSuite(carto.test_suite || 'auto')

      const detectedProtectedCols: string[] = carto.detected_protected_cols?.length
        ? carto.detected_protected_cols
        : ['auto']
      const detectedTargetCol: string = carto.detected_target_col || 'auto'
      const resolvedTestSuite: string = carto.test_suite || 'auto'

      store.setProtectedCols(detectedProtectedCols)
      store.setTargetCol(detectedTargetCol)
      setProgress(40)

      setRunningStage('Counterfactual Constitution')
      store.setStage('constitution')
      const constitution = await runConstitution(
        store.modelFile,
        detectedProtectedCols,
        detectedTargetCol,
        carto,
        modelType,
        modelEndpoint,
        resolvedLlmKey,
        hfToken,
        resolvedTestSuite,
      )
      store.setConstitutionResults(constitution); setProgress(70)

      setRunningStage('Proxy Variable Hunter')
      store.setStage('proxy')
      const proxy = await runProxyHunter(
        detectedProtectedCols,
        detectedTargetCol,
        modelType,
        resolvedTestSuite,
      )
      store.setProxyResults(proxy); setProgress(100)

      store.setStage('review')
      nav('/results')
    } catch (e: any) {
      store.setError(e.message)
    } finally {
      store.setLoading(false)
      setRunningStage(null)
    }
  }

  return (
    <div className="max-w-3xl mx-auto px-6 py-12">
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mb-10">
        <div className="text-xs font-mono text-lens-light mb-2">Step 1 of 6</div>
        <h1 className="font-display font-bold text-white text-3xl mb-2">Configure Audit</h1>
        <p className="text-white/40 text-sm">
          Connect your model. FairLens runs it against curated bias test datasets and tells you
          exactly what biases exist — no data upload needed.
        </p>
      </motion.div>

      {/* ── Built-in dataset notice ──────────────────────────────────────── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}
        className="glass rounded-2xl p-4 border border-lens/10 mb-6 flex items-start gap-3">
        <div className="text-lens text-xl flex-shrink-0">🗂</div>
        <div>
          <div className="text-lens-light font-mono text-xs font-semibold mb-1">Built-in test suite</div>
          <div className="text-white/40 text-xs leading-relaxed">
            <span className="text-white/60 font-semibold">{SUITE_INFO[autoSuite].label}</span>
            {' — '}{SUITE_INFO[autoSuite].detail}
          </div>
          <div className="text-white/25 text-xs mt-1">
            Suite auto-selects based on model type. Both suites ship with the repo — no internet required.
          </div>
        </div>
      </motion.div>

      {/* ── Model Type ──────────────────────────────────────────────────── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="mb-6">
        <h3 className="text-xs font-mono text-white/40 uppercase tracking-widest mb-3">Model</h3>
        <div className="grid grid-cols-2 gap-2 mb-4">
          {MODEL_TYPES.map(t => (
            <button key={t.id} onClick={() => setModelType(t.id)}
              className={`text-left p-3 rounded-xl border transition-all
                ${modelType === t.id ? 'border-lens/50 bg-lens/10' : 'border-white/8 hover:border-white/15'}`}>
              <div className="text-base mb-1">{t.icon}</div>
              <div className="text-xs font-mono font-semibold text-white mb-0.5">{t.label}</div>
              <div className="text-xs text-white/30">{t.desc}</div>
            </button>
          ))}
        </div>

        <AnimatePresence mode="wait">
          <motion.div key={modelType} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            {modelType === 'sklearn' && (
              <DropZone file={store.modelFile} onDrop={onDropModel} />
            )}
            {modelType === 'api' && (
              <input value={apiEndpoint} onChange={e => setApiEndpoint(e.target.value)}
                placeholder="https://my-model-api.com"
                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono" />
            )}
            {modelType === 'huggingface' && (
              <div className="space-y-3">
                <input value={hfModel} onChange={e => setHfModel(e.target.value)}
                  placeholder="e.g. unitary/toxic-bert or cardiffnlp/twitter-roberta-base-sentiment"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono" />
                <input value={hfToken} onChange={e => setHfToken(e.target.value)}
                  placeholder="HuggingFace token (hf_...) — required for most models"
                  type="password"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono" />
                <p className="text-white/25 text-xs">Model type (classifier or generative) is auto-detected from the HuggingFace Hub.</p>
              </div>
            )}
            {modelType === 'openai' && (
              <div className="space-y-3">
                <input value={openaiModel} onChange={e => setOpenaiModel(e.target.value)}
                  placeholder="e.g. gpt-4o, gpt-4, gpt-3.5-turbo"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono" />
                <input value={openaiKey} onChange={e => setOpenaiKey(e.target.value)}
                  placeholder="OpenAI API key (sk-...)"
                  type="password"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono" />
              </div>
            )}
            {modelType === 'gemini_llm' && (
              <div className="space-y-3">
                <input value={geminiModel} onChange={e => setGeminiModel(e.target.value)}
                  placeholder="e.g. gemini-2.0-flash, gemini-1.5-pro"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono" />
                <input value={geminiKey} onChange={e => setGeminiKey(e.target.value)}
                  placeholder="Gemini API key (AIza...)"
                  type="password"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono" />
              </div>
            )}
            {modelType === 'vertex_ai' && (
              <div className="space-y-3">
                <input value={vertexEndpoint} onChange={e => setVertexEndpoint(e.target.value)}
                  placeholder="Vertex AI Endpoint ID"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono" />
                <input value={gcpProject} onChange={e => setGcpProject(e.target.value)}
                  placeholder="GCP Project ID"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono" />
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </motion.div>

      {/* ── Auto-detect notice ───────────────────────────────────────────── */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
        className="glass rounded-2xl p-4 border border-lens/10 mb-8 flex items-start gap-3">
        <div className="text-lens text-xl flex-shrink-0">✦</div>
        <div>
          <div className="text-lens-light font-mono text-xs font-semibold mb-1">Auto-detection enabled</div>
          <div className="text-white/40 text-xs leading-relaxed">
            FairLens automatically detects protected attributes (gender, race, religion, etc.),
            identifies the prediction target, and surfaces every bias it finds across demographic groups.
            No manual configuration needed.
          </div>
        </div>
      </motion.div>

      {/* Error */}
      <AnimatePresence>
        {store.error && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="bg-signal-red/10 border border-signal-red/30 rounded-xl p-4 mb-6 text-signal-red text-sm font-mono">
            ⚠ {store.error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading */}
      <AnimatePresence>
        {store.loading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="glass rounded-2xl p-5 border border-lens/20 mb-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-2 h-2 rounded-full bg-lens animate-pulse" />
              <span className="text-lens-light font-mono text-sm">Running {runningStage} via Gemini 2.5 Flash...</span>
            </div>
            <div className="w-full bg-white/5 rounded-full h-1.5">
              <motion.div className="h-1.5 rounded-full bg-gradient-to-r from-lens to-lens-light"
                animate={{ width: `${progress}%` }} transition={{ duration: 0.5 }} />
            </div>
            <div className="text-white/25 text-xs font-mono mt-2">{progress}% — running model against built-in bias probes</div>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.button
        whileHover={canRun && !store.loading ? { scale: 1.01 } : {}}
        whileTap={canRun && !store.loading ? { scale: 0.99 } : {}}
        disabled={!canRun || store.loading}
        onClick={runAudit}
        className={`w-full py-4 rounded-xl font-display font-semibold text-base transition-all
          ${canRun && !store.loading
            ? 'bg-lens hover:bg-lens/90 text-white glow-lens cursor-pointer'
            : 'bg-white/5 text-white/20 cursor-not-allowed'}`}>
        {store.loading ? 'Analysing...' : 'Detect Biases Automatically →'}
      </motion.button>

      <p className="text-center text-white/20 text-xs mt-3 font-mono">
        Powered by Gemini 2.5 Flash · Built-in bias test datasets · No data upload required
      </p>
    </div>
  )
}
