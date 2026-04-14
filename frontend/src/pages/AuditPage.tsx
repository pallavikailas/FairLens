import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuditStore } from '../hooks/useAuditStore'
import { runCartography, runConstitution, runProxyHunter, type ModelType } from '../utils/api'

// ── Model type definitions ─────────────────────────────────────────────────
const MODEL_TYPES: { id: ModelType; label: string; icon: string; desc: string; fields: string[] }[] = [
  {
    id: 'sklearn',
    label: 'scikit-learn / XGBoost / LightGBM',
    icon: '🧠',
    desc: 'Any sklearn-compatible .pkl model — RandomForest, XGBoost, CatBoost, LogisticRegression, SVM, Pipelines…',
    fields: ['model_file'],
  },
  {
    id: 'api',
    label: 'REST API Endpoint',
    icon: '🌐',
    desc: 'Any model served behind an HTTP endpoint. FairLens POSTs rows as JSON and reads predictions.',
    fields: ['api_endpoint'],
  },
  {
    id: 'huggingface',
    label: 'HuggingFace Model',
    icon: '🤗',
    desc: 'Any text-classification pipeline from HuggingFace Hub. Provide the model name or path.',
    fields: ['api_endpoint'],
  },
  {
    id: 'vertex_ai',
    label: 'Vertex AI Endpoint',
    icon: '☁',
    desc: 'A model deployed on Google Cloud Vertex AI. Provide the endpoint ID and GCP project.',
    fields: ['vertex_endpoint_id', 'gcp_project'],
  },
]

// ── Reusable dropzone ──────────────────────────────────────────────────────
function DropZone({ label, accept, file, onDrop, icon }: any) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept, maxFiles: 1 })
  return (
    <div
      {...getRootProps()}
      className={`relative border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all
        ${isDragActive ? 'border-lens bg-lens/5' : file ? 'border-signal-green/40 bg-signal-green/5' : 'border-white/10 hover:border-white/20'}`}
    >
      <input {...getInputProps()} />
      <div className="text-2xl mb-2">{file ? '✓' : icon}</div>
      {file ? (
        <div>
          <div className="text-signal-green font-mono text-sm">{file.name}</div>
          <div className="text-white/30 text-xs mt-1">{(file.size / 1024).toFixed(1)} KB</div>
        </div>
      ) : (
        <div>
          <div className="text-white/60 text-sm font-medium mb-1">{label}</div>
          <div className="text-white/25 text-xs">{isDragActive ? 'Drop here' : 'Drag & drop or click'}</div>
        </div>
      )}
    </div>
  )
}

function ColTag({ col, onRemove }: { col: string; onRemove?: () => void }) {
  return (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-lens/10 border border-lens/30 text-lens-light text-xs font-mono">
      {col}
      {onRemove && <button onClick={onRemove} className="hover:opacity-70 ml-1">×</button>}
    </span>
  )
}

export default function AuditPage() {
  const nav = useNavigate()
  const store = useAuditStore()
  const [newCol, setNewCol] = useState('')
  const [runningStage, setRunningStage] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)

  const cfg = store.modelConfig

  const onDropDataset = useCallback((files: File[]) => {
    if (files[0]) store.setDatasetFile(files[0])
  }, [])

  const onDropModel = useCallback((files: File[]) => {
    if (files[0]) store.setModelConfig({ modelFile: files[0] })
  }, [])

  const addProtectedCol = () => {
    const col = newCol.trim()
    if (col && !store.protectedCols.includes(col)) {
      store.setProtectedCols([...store.protectedCols, col])
    }
    setNewCol('')
  }

  const selectedType = MODEL_TYPES.find(t => t.id === cfg.modelType)!

  const canRun = (() => {
    if (!store.datasetFile || store.protectedCols.length === 0 || !store.targetCol) return false
    if (cfg.modelType === 'sklearn' && !cfg.modelFile) return false
    if (cfg.modelType === 'api' && !cfg.apiEndpoint) return false
    if (cfg.modelType === 'huggingface' && !cfg.apiEndpoint) return false
    if (cfg.modelType === 'vertex_ai' && (!cfg.vertexEndpointId || !cfg.gcpProject)) return false
    return true
  })()

  const runFullPipeline = async () => {
    if (!canRun) return
    store.setError(null)
    store.setLoading(true)

    try {
      setRunningStage('Bias Cartography')
      setProgress(10)
      store.setStage('cartography')
      const carto = await runCartography(cfg, store.datasetFile!, store.protectedCols, store.targetCol)
      store.setCartographyResults(carto)
      setProgress(40)

      setRunningStage('Counterfactual Constitution')
      store.setStage('constitution')
      const constitution = await runConstitution(cfg, store.datasetFile!, store.protectedCols, store.targetCol, carto)
      store.setConstitutionResults(constitution)
      setProgress(70)

      setRunningStage('Proxy Variable Hunter')
      store.setStage('proxy')
      const proxy = await runProxyHunter(store.datasetFile!, store.protectedCols, store.targetCol)
      store.setProxyResults(proxy)
      setProgress(100)

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
          FairLens works as a plugin for any model. Select your model type, upload your dataset, and specify protected attributes.
        </p>
      </motion.div>

      {/* Model type selector */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}
        className="mb-6">
        <h3 className="text-xs font-mono text-white/40 uppercase tracking-widest mb-3">Model Type</h3>
        <div className="grid grid-cols-2 gap-3">
          {MODEL_TYPES.map((t) => (
            <button
              key={t.id}
              onClick={() => store.setModelConfig({ modelType: t.id, modelFile: undefined, apiEndpoint: undefined, vertexEndpointId: undefined, gcpProject: undefined })}
              className={`text-left p-4 rounded-2xl border transition-all
                ${cfg.modelType === t.id
                  ? 'border-lens/50 bg-lens/10 text-white'
                  : 'border-white/8 bg-white/2 hover:border-white/15 text-white/50'}`}
            >
              <div className="text-xl mb-2">{t.icon}</div>
              <div className="text-xs font-mono font-semibold mb-1">{t.label}</div>
              <div className="text-xs opacity-60 leading-relaxed">{t.desc}</div>
            </button>
          ))}
        </div>
      </motion.div>

      {/* Model-type-specific fields */}
      <AnimatePresence mode="wait">
        <motion.div key={cfg.modelType}
          initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
          className="glass rounded-2xl p-5 border border-white/5 mb-6">
          <div className="flex items-center gap-2 mb-4">
            <span className="text-lens">{selectedType.icon}</span>
            <h3 className="font-display font-semibold text-white text-sm">{selectedType.label}</h3>
          </div>

          {cfg.modelType === 'sklearn' && (
            <DropZone label="Trained Model (.pkl)" accept={{ 'application/octet-stream': ['.pkl'] }}
              file={cfg.modelFile} onDrop={onDropModel} icon="🧠" />
          )}

          {(cfg.modelType === 'api') && (
            <div>
              <label className="text-white/40 text-xs font-mono mb-2 block">REST API Endpoint URL</label>
              <input
                value={cfg.apiEndpoint || ''}
                onChange={e => store.setModelConfig({ apiEndpoint: e.target.value })}
                placeholder="https://my-model-api.com"
                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono"
              />
              <p className="text-white/25 text-xs mt-2">
                FairLens sends: POST /predict with body <code className="text-lens-light">{'{"instances": [[...]]}'}</code><br/>
                Expected response: <code className="text-lens-light">{'{"predictions": [...], "probabilities": [[...]]}'}</code>
              </p>
            </div>
          )}

          {cfg.modelType === 'huggingface' && (
            <div>
              <label className="text-white/40 text-xs font-mono mb-2 block">HuggingFace Model Name</label>
              <input
                value={cfg.apiEndpoint || ''}
                onChange={e => store.setModelConfig({ apiEndpoint: e.target.value })}
                placeholder="distilbert-base-uncased-finetuned-sst-2-english"
                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono"
              />
              <p className="text-white/25 text-xs mt-2">
                Your dataset must contain a <code className="text-lens-light">text</code> column for HuggingFace models.
              </p>
            </div>
          )}

          {cfg.modelType === 'vertex_ai' && (
            <div className="space-y-3">
              <div>
                <label className="text-white/40 text-xs font-mono mb-2 block">Vertex AI Endpoint ID</label>
                <input
                  value={cfg.vertexEndpointId || ''}
                  onChange={e => store.setModelConfig({ vertexEndpointId: e.target.value })}
                  placeholder="1234567890123456789"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono"
                />
              </div>
              <div>
                <label className="text-white/40 text-xs font-mono mb-2 block">GCP Project ID</label>
                <input
                  value={cfg.gcpProject || ''}
                  onChange={e => store.setModelConfig({ gcpProject: e.target.value })}
                  placeholder="my-gcp-project-id"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono"
                />
              </div>
            </div>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Dataset upload */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}
        className="mb-6">
        <DropZone label="Dataset (.csv)" accept={{ 'text/csv': ['.csv'] }}
          file={store.datasetFile} onDrop={onDropDataset} icon="📊" />
      </motion.div>

      {/* Protected cols */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
        className="glass rounded-2xl p-5 border border-white/5 mb-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-display font-semibold text-white text-sm">Protected Attribute Columns</h3>
          <button
            onClick={() => { store.setProtectedCols(['gender', 'race', 'age_group']); store.setTargetCol('hired') }}
            className="text-xs font-mono text-lens-light border border-lens/20 px-3 py-1 rounded-full hover:bg-lens/10 transition-all"
          >
            Load demo
          </button>
        </div>
        <div className="flex flex-wrap gap-2 mb-3 min-h-[28px]">
          {store.protectedCols.map(col => (
            <ColTag key={col} col={col} onRemove={() =>
              store.setProtectedCols(store.protectedCols.filter(c => c !== col))} />
          ))}
          {store.protectedCols.length === 0 && <span className="text-white/20 text-xs font-mono">None added yet</span>}
        </div>
        <div className="flex gap-2">
          <input value={newCol} onChange={e => setNewCol(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && addProtectedCol()}
            placeholder="Column name (e.g. gender)"
            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono"
          />
          <button onClick={addProtectedCol}
            className="bg-lens/20 hover:bg-lens/30 border border-lens/30 text-lens-light px-4 rounded-xl text-sm transition-all">
            Add
          </button>
        </div>
      </motion.div>

      {/* Target column */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}
        className="glass rounded-2xl p-5 border border-white/5 mb-8">
        <h3 className="font-display font-semibold text-white text-sm mb-2">Target Column</h3>
        <input value={store.targetCol} onChange={e => store.setTargetCol(e.target.value)}
          placeholder="e.g. hired"
          className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder-white/20 focus:outline-none focus:border-signal-green/50 font-mono"
        />
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

      {/* Loading bar */}
      <AnimatePresence>
        {store.loading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="glass rounded-2xl p-5 border border-lens/20 mb-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-2 h-2 rounded-full bg-lens animate-pulse" />
              <span className="text-lens-light font-mono text-sm">Running {runningStage}...</span>
            </div>
            <div className="w-full bg-white/5 rounded-full h-1.5">
              <motion.div className="h-1.5 rounded-full bg-gradient-to-r from-lens to-lens-light"
                animate={{ width: `${progress}%` }} transition={{ duration: 0.5 }} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.button
        whileHover={canRun && !store.loading ? { scale: 1.01 } : {}}
        whileTap={canRun && !store.loading ? { scale: 0.99 } : {}}
        disabled={!canRun || store.loading}
        onClick={runFullPipeline}
        className={`w-full py-4 rounded-xl font-display font-semibold text-base transition-all
          ${canRun && !store.loading ? 'bg-lens hover:bg-lens/90 text-white glow-lens cursor-pointer' : 'bg-white/5 text-white/20 cursor-not-allowed'}`}
      >
        {store.loading ? 'Analysing...' : `Run Full Bias Audit — ${selectedType.label} →`}
      </motion.button>
    </div>
  )
}
