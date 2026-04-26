import { useState, useCallback, useRef, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuditStore } from '../hooks/useAuditStore'
import { runModelProbe, runDatasetProbe, runCrossAnalysis } from '../utils/api'

// ── Model type config ──────────────────────────────────────────────────────
const MODEL_TYPES = [
  { id: 'sklearn',     icon: '🧠', label: 'scikit-learn / XGBoost',      desc: 'Upload a .pkl model file' },
  { id: 'api',         icon: '🌐', label: 'REST API Endpoint',            desc: 'Any model behind an HTTP URL' },
  { id: 'huggingface', icon: '🤗', label: 'HuggingFace',                  desc: 'Any model from HF Hub — auto-detected' },
  { id: 'openai',      icon: '🔮', label: 'OpenAI (ChatGPT / GPT-4)',     desc: 'gpt-4o, gpt-4, gpt-3.5-turbo' },
  { id: 'gemini_llm',  icon: '✦',  label: 'Gemini LLM',                  desc: 'gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash' },
  { id: 'vertex_ai',   icon: '☁',  label: 'Vertex AI Endpoint',           desc: 'Google Cloud deployed model' },
] as const

type ModelType = typeof MODEL_TYPES[number]['id']

// ── Dataset source config ──────────────────────────────────────────────────
const DATASET_SOURCES = [
  { id: 'upload',      icon: '📂', label: 'Upload CSV',          desc: 'Upload from your device' },
  { id: 'url',         icon: '🔗', label: 'URL',                 desc: 'Direct link to a CSV file' },
  { id: 'huggingface', icon: '🤗', label: 'HuggingFace Dataset', desc: 'Dataset name from HuggingFace Hub' },
  { id: 'kaggle',      icon: '📊', label: 'Kaggle Dataset',      desc: 'Kaggle dataset path (owner/dataset)' },
] as const

type DatasetSource = typeof DATASET_SOURCES[number]['id']

// ── Pipeline phases for progress display ────────────────────────────────────
const PHASES = [
  {
    id: 'model_probe',
    label: 'Model Probe',
    desc: 'Checking model for hidden biases on reference dataset',
    icon: '🔬',
  },
  {
    id: 'dataset_probe',
    label: 'Dataset Analysis',
    desc: 'Analyzing dataset for structural biases & proxy chains',
    icon: '📊',
  },
  {
    id: 'cross_analysis',
    label: 'Cross-Analysis',
    desc: 'Grouping model × dataset biases — finding interaction risks',
    icon: '⚡',
  },
  {
    id: 'review',
    label: 'Review & Red-Team',
    desc: 'Confirm findings then run adversarial remediation',
    icon: '🎯',
  },
]

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

function DatasetDropZone({ file, onDrop }: { file: File | null; onDrop: (f: File[]) => void }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'text/csv': ['.csv'] }, maxFiles: 1
  })
  return (
    <div {...getRootProps()} className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all
      ${isDragActive ? 'border-lens bg-lens/5' : file ? 'border-signal-green/40 bg-signal-green/5' : 'border-white/10 hover:border-white/20'}`}>
      <input {...getInputProps()} />
      <div className="text-3xl mb-2">{file ? '✓' : '📊'}</div>
      {file ? (
        <div>
          <div className="text-signal-green font-mono text-sm">{file.name}</div>
          <div className="text-white/30 text-xs mt-1">{(file.size / 1024).toFixed(1)} KB</div>
        </div>
      ) : (
        <div>
          <div className="text-white/60 text-sm font-medium mb-1">Drop your CSV here</div>
          <div className="text-white/30 text-xs">{isDragActive ? 'Drop it!' : 'or click to browse'}</div>
        </div>
      )}
    </div>
  )
}

export default function AuditPage() {
  const nav   = useNavigate()
  const store = useAuditStore()

  const [modelType, setModelType]       = useState<ModelType>('sklearn')
  const [datasetSource, setDatasetSource] = useState<DatasetSource>('upload')
  const [apiEndpoint, setApiEndpoint]   = useState('')
  const [hfModel, setHfModel]           = useState('')
  const [hfToken, setHfToken]           = useState('')
  const [openaiModel, setOpenaiModel]   = useState('gpt-4o')
  const [openaiKey, setOpenaiKey]       = useState('')
  const [geminiModel, setGeminiModel]   = useState('gemini-2.5-flash')
  const [geminiKey, setGeminiKey]       = useState('')
  const [vertexEndpoint, setVertexEndpoint] = useState('')
  const [gcpProject, setGcpProject]     = useState('')
  const [datasetUrl, setDatasetUrl]     = useState('')
  const [hfDataset, setHfDataset]       = useState('')
  const [kaggleDataset, setKaggleDataset] = useState('')
  const [activePhase, setActivePhase]   = useState<string | null>(null)
  const [completedPhases, setCompletedPhases] = useState<string[]>([])
  const [progress, setProgress]         = useState(0)
  const [subStage, setSubStage]         = useState<string | null>(null)
  const subStageTimers = useRef<ReturnType<typeof setTimeout>[]>([])

  useEffect(() => () => subStageTimers.current.forEach(clearTimeout), [])

  const scheduleSubStages = (stages: { name: string; delay: number }[]) => {
    subStageTimers.current.forEach(clearTimeout)
    subStageTimers.current = stages.map(({ name, delay }) =>
      setTimeout(() => { setSubStage(name); store.setActiveSubStage(name) }, delay)
    )
  }
  const clearSubStages = () => {
    subStageTimers.current.forEach(clearTimeout)
    subStageTimers.current = []
    setSubStage(null)
    store.setActiveSubStage(null)
  }

  const PHASE_SUB_STAGES: Record<string, string[]> = {
    model_probe:    ['Cartography', 'Constitution'],
    dataset_probe:  ['Cartography', 'Proxy Hunt'],
    cross_analysis: ['Cartography', 'Constitution', 'Proxy Hunt'],
  }

  const onDropModel   = useCallback((files: File[]) => { if (files[0]) store.setModelFile(files[0]) }, [])
  const onDropDataset = useCallback((files: File[]) => { if (files[0]) store.setDatasetFile(files[0]) }, [])

  const datasetReady = datasetSource === 'upload'
    ? !!store.datasetFile
    : datasetSource === 'url'        ? !!datasetUrl
    : datasetSource === 'huggingface' ? !!hfDataset
    : !!kaggleDataset

  const hasModel = modelType === 'sklearn'
    ? !!store.modelFile
    : !!(modelType === 'huggingface' ? hfModel
       : modelType === 'openai'      ? openaiModel
       : modelType === 'gemini_llm'  ? geminiModel
       : apiEndpoint)

  const canRun = hasModel || datasetReady

  const runAudit = async () => {
    if (!canRun) return
    store.setError(null)
    store.setLoading(true)
    setCompletedPhases([])

    try {
      const resolvedDatasetUrl = datasetUrl || hfDataset || kaggleDataset || ''

      store.setDatasetSource(datasetSource)
      store.setDatasetUrl(resolvedDatasetUrl)

      const modelEndpoint =
        modelType === 'huggingface' ? hfModel :
        modelType === 'openai'      ? openaiModel :
        modelType === 'gemini_llm'  ? geminiModel :
        apiEndpoint

      const resolvedLlmKey =
        modelType === 'openai'     ? openaiKey :
        modelType === 'gemini_llm' ? geminiKey : ''

      store.setModelType(modelType)
      store.setModelEndpoint(modelEndpoint)
      store.setLlmApiKey(resolvedLlmKey)
      store.setHfToken(hfToken)

      // ── Phase 1: Model Probe (embedded reference dataset) ─────────────────
      let modelProbe: any = null
      if (hasModel) {
        setActivePhase('model_probe')
        store.setStage('model_probe')
        setProgress(5)
        scheduleSubStages([
          { name: 'Cartography', delay: 0 },
          { name: 'Constitution', delay: 18000 },
        ])

        modelProbe = await runModelProbe(
          store.modelFile,
          modelType,
          modelEndpoint,
          resolvedLlmKey,
          hfToken,
        )
        clearSubStages()
        store.setModelProbeResults(modelProbe)
        setCompletedPhases(p => [...p, 'model_probe'])
        setProgress(datasetReady ? 30 : 100)
      }

      // ── Phase 2 & 3: only when a dataset is provided ──────────────────────
      if (datasetReady) {
        // ── Phase 2: Dataset Probe (user dataset, no model) ─────────────────
        setActivePhase('dataset_probe')
        store.setStage('dataset_probe')
        setProgress(35)
        scheduleSubStages([
          { name: 'Cartography', delay: 0 },
          { name: 'Proxy Hunt', delay: 12000 },
        ])

        const datasetProbe = await runDatasetProbe(
          store.datasetFile,
          ['auto'],
          'auto',
          datasetSource,
          resolvedDatasetUrl,
        )
        clearSubStages()
        store.setDatasetProbeResults(datasetProbe)

        const detectedProtectedCols: string[] = datasetProbe.detected_protected_cols?.length
          ? datasetProbe.detected_protected_cols
          : ['auto']
        const detectedTargetCol: string = datasetProbe.detected_target_col || 'auto'
        store.setProtectedCols(detectedProtectedCols)
        store.setTargetCol(detectedTargetCol)

        setCompletedPhases(p => [...p, 'dataset_probe'])
        setProgress(65)

        // ── Phase 3: Cross-Analysis (model × user dataset) ────────────────
        // Only runs when a model is also available
        if (hasModel && modelProbe) {
          setActivePhase('cross_analysis')
          store.setStage('cross_analysis')
          scheduleSubStages([
            { name: 'Cartography', delay: 0 },
            { name: 'Constitution', delay: 15000 },
            { name: 'Proxy Hunt', delay: 40000 },
          ])

          const crossAnalysis = await runCrossAnalysis(
            modelProbe,
            datasetProbe,
            store.modelFile,
            store.datasetFile,
            modelType,
            modelEndpoint,
            resolvedLlmKey,
            hfToken,
            datasetSource,
            resolvedDatasetUrl,
            detectedProtectedCols,
            detectedTargetCol,
          )
          clearSubStages()
          store.setCrossAnalysisResults(crossAnalysis)

          store.setCartographyResults(crossAnalysis.cartography)
          store.setConstitutionResults(crossAnalysis.constitution)
          store.setProxyResults(crossAnalysis.proxy)

          setCompletedPhases(p => [...p, 'cross_analysis'])
        }

        setProgress(100)
      }

      store.setStage('review')
      nav('/results')
    } catch (e: any) {
      store.setError(e.message)
    } finally {
      store.setLoading(false)
      setActivePhase(null)
    }
  }

  return (
    <div className="max-w-3xl mx-auto px-6 py-12">
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mb-10">
        <div className="text-xs font-mono text-lens-light mb-2">Step 1 of 4</div>
        <h1 className="font-display font-bold text-white text-3xl mb-2">Configure Audit</h1>
        <p className="text-white/40 text-sm">
          Connect your model and/or dataset. FairLens runs up to 3 phases — phases that require
          a missing input are automatically skipped.
        </p>
      </motion.div>

      {/* ── Pipeline phases preview ────────────────────────────────────────── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.02 }}
        className="grid grid-cols-2 gap-2 mb-8">
        {PHASES.map((phase, i) => {
          const done    = completedPhases.includes(phase.id)
          const running = activePhase === phase.id
          const phaseSubs = PHASE_SUB_STAGES[phase.id] ?? []
          const needsDataset = phase.id === 'dataset_probe' || phase.id === 'cross_analysis'
          const needsModel   = phase.id === 'model_probe' || phase.id === 'cross_analysis'
          const willSkip = (needsDataset && !datasetReady) || (needsModel && !hasModel && !datasetReady)
          return (
            <div key={phase.id} className={`rounded-xl p-3 border transition-all
              ${running  ? 'border-lens/50 bg-lens/10' :
                done     ? 'border-signal-green/30 bg-signal-green/5' :
                willSkip ? 'border-white/5 bg-white/1 opacity-40' :
                           'border-white/8 bg-white/2'}`}>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-base">{done ? '✓' : willSkip ? '—' : phase.icon}</span>
                <span className={`text-xs font-mono font-semibold
                  ${running ? 'text-lens-light' : done ? 'text-signal-green' : willSkip ? 'text-white/25' : 'text-white/50'}`}>
                  Phase {i + 1} — {phase.label}
                  {willSkip && <span className="ml-2 text-white/20 normal-case font-normal">(skipped — no {needsDataset && !datasetReady ? 'dataset' : 'model'})</span>}
                </span>
              </div>
              <p className="text-white/30 text-xs leading-relaxed">{phase.desc}</p>
              {running && phaseSubs.length > 0 && (
                <div className="flex gap-1 mt-2 flex-wrap">
                  {phaseSubs.map(sub => (
                    <span key={sub} className={`text-xs px-1.5 py-0.5 rounded font-mono transition-all
                      ${subStage === sub
                        ? 'bg-lens/25 text-lens-light border border-lens/40'
                        : 'text-white/20 border border-white/8'}`}>
                      {subStage === sub && <span className="inline-block animate-pulse mr-1">▶</span>}
                      {sub}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </motion.div>

      {/* ── Dataset Source ──────────────────────────────────────────────────── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }} className="mb-6">
        <h3 className="text-xs font-mono text-white/40 uppercase tracking-widest mb-3">Dataset Source</h3>
        <div className="grid grid-cols-2 gap-2 mb-4">
          {DATASET_SOURCES.map(s => (
            <button key={s.id} onClick={() => setDatasetSource(s.id)}
              className={`text-left p-3 rounded-xl border transition-all
                ${datasetSource === s.id ? 'border-lens/50 bg-lens/10' : 'border-white/8 hover:border-white/15'}`}>
              <div className="text-base mb-1">{s.icon}</div>
              <div className="text-xs font-mono font-semibold text-white mb-0.5">{s.label}</div>
              <div className="text-xs text-white/30">{s.desc}</div>
            </button>
          ))}
        </div>

        <AnimatePresence mode="wait">
          <motion.div key={datasetSource} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            {datasetSource === 'upload' && (
              <DatasetDropZone file={store.datasetFile} onDrop={onDropDataset} />
            )}
            {datasetSource === 'url' && (
              <div>
                <input value={datasetUrl} onChange={e => setDatasetUrl(e.target.value)}
                  placeholder="https://example.com/dataset.csv"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono" />
                <p className="text-white/25 text-xs mt-2">Must be a direct link to a publicly accessible CSV file.</p>
              </div>
            )}
            {datasetSource === 'huggingface' && (
              <div>
                <input value={hfDataset} onChange={e => setHfDataset(e.target.value)}
                  placeholder="e.g. csv/csv or Rowan/hellaswag"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono" />
                <p className="text-white/25 text-xs mt-2">Dataset name from <span className="text-lens-light">huggingface.co/datasets</span></p>
              </div>
            )}
            {datasetSource === 'kaggle' && (
              <div>
                <input value={kaggleDataset} onChange={e => setKaggleDataset(e.target.value)}
                  placeholder="e.g. dsinghania25/hiring-bias-dataset"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono" />
                <p className="text-white/25 text-xs mt-2">Owner/dataset-name from <span className="text-lens-light">kaggle.com/datasets</span></p>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </motion.div>

      {/* ── Model Type ──────────────────────────────────────────────────────── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="mb-6">
        <h3 className="text-xs font-mono text-white/40 uppercase tracking-widest mb-3">
          Model Type <span className="text-white/20 normal-case">(required — Phase 1 probes model on embedded reference dataset)</span>
        </h3>
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
                  placeholder="e.g. unitary/toxic-bert or google/gemma-3-1b-it"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono" />
                <input value={hfToken} onChange={e => setHfToken(e.target.value)}
                  placeholder="HuggingFace token (hf_...) — required for most models"
                  type="password"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono" />
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
                  placeholder="e.g. gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash"
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

      {/* ── Pipeline explanation ─────────────────────────────────────────────── */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
        className="glass rounded-2xl p-4 border border-lens/10 mb-8 flex items-start gap-3">
        <div className="text-lens text-xl flex-shrink-0">✦</div>
        <div>
          <div className="text-lens-light font-mono text-xs font-semibold mb-1">Adaptive Bias Analysis</div>
          <div className="text-white/40 text-xs leading-relaxed space-y-1">
            <div><span className="text-white/60">Phase 1 — Model Probe:</span> Your model is tested against a neutral embedded reference dataset to expose hidden intrinsic biases. <span className="text-white/30 italic">Requires a model.</span></div>
            <div><span className="text-white/60">Phase 2 — Dataset Analysis:</span> Your dataset is analyzed without the model — cartography + proxy chains reveal structural biases. <span className="text-white/30 italic">Requires a dataset.</span></div>
            <div><span className="text-white/60">Phase 3 — Cross-Analysis:</span> Model × dataset interaction biases. Compounded risks and blind spots surfaced. <span className="text-white/30 italic">Requires both.</span></div>
            <div className="pt-1 text-white/25">Phases are skipped automatically when their required input is not provided.</div>
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
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            className="glass rounded-2xl border border-lens/20 mb-6 overflow-hidden">
            {/* Header bar */}
            <div className="flex items-center gap-3 px-5 py-3 border-b border-white/5 bg-lens/5">
              <motion.div className="w-2 h-2 rounded-full bg-lens"
                animate={{ opacity: [1, 0.3, 1] }} transition={{ duration: 1.2, repeat: Infinity }} />
              <span className="text-lens-light font-mono text-sm font-semibold">
                {activePhase === 'model_probe'    ? 'Phase 1 — Model Probe' :
                 activePhase === 'dataset_probe'  ? 'Phase 2 — Dataset Analysis' :
                 activePhase === 'cross_analysis' ? 'Phase 3 — Cross-Analysis' :
                 'Initialising…'}
              </span>
              {subStage && (
                <span className="ml-auto text-xs font-mono px-2.5 py-1 rounded-lg bg-lens/20 text-lens-light border border-lens/30">
                  ▶ {subStage}
                </span>
              )}
            </div>

            {/* Per-phase sub-stage track */}
            {activePhase && PHASE_SUB_STAGES[activePhase] && (
              <div className="flex gap-0 border-b border-white/5">
                {PHASE_SUB_STAGES[activePhase].map((sub, i, arr) => {
                  const isActive = subStage === sub
                  const isDone   = arr.indexOf(subStage ?? '') > i
                  return (
                    <div key={sub} className={`flex-1 flex items-center gap-2 px-4 py-2.5 transition-all
                      ${isActive ? 'bg-lens/10' : isDone ? 'bg-signal-green/5' : 'bg-transparent'}`}>
                      <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs flex-shrink-0
                        ${isActive ? 'bg-lens/30 border border-lens/50' : isDone ? 'bg-signal-green/20 border border-signal-green/30' : 'border border-white/10'}`}>
                        {isDone ? <span className="text-signal-green text-xs">✓</span> :
                         isActive ? <motion.span animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }} className="text-lens-light inline-block text-xs">⟳</motion.span> :
                         <span className="text-white/20 text-xs">{i + 1}</span>}
                      </div>
                      <span className={`text-xs font-mono font-semibold
                        ${isActive ? 'text-lens-light' : isDone ? 'text-signal-green' : 'text-white/25'}`}>
                        {sub}
                      </span>
                    </div>
                  )
                })}
              </div>
            )}

            {/* Progress bar */}
            <div className="px-5 py-3">
              <div className="w-full bg-white/5 rounded-full h-1">
                <motion.div className="h-1 rounded-full bg-gradient-to-r from-lens to-lens-light"
                  animate={{ width: `${progress}%` }} transition={{ duration: 0.6, ease: 'easeOut' }} />
              </div>
              <div className="text-white/25 text-xs font-mono mt-1.5">{progress}% complete</div>
            </div>
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
        {store.loading ? 'Analysing...' : !datasetReady && hasModel ? 'Run Model-Only Analysis →' : 'Run 3-Phase Bias Analysis →'}
      </motion.button>

      <p className="text-center text-white/20 text-xs mt-3 font-mono">
        Powered by Gemini 2.5 Flash · Google Cloud · Auto-detects protected attributes
      </p>
    </div>
  )
}
