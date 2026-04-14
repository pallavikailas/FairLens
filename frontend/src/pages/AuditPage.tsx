import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuditStore } from '../hooks/useAuditStore'
import { runCartography, runConstitution, runProxyHunter } from '../utils/api'

function DropZone({ label, accept, file, onDrop, icon }: any) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept, maxFiles: 1 })
  return (
    <div {...getRootProps()} className={`
      border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all
      ${isDragActive ? 'border-lens bg-lens/5' : file
        ? 'border-signal-green/40 bg-signal-green/5'
        : 'border-white/10 hover:border-white/20'}`}>
      <input {...getInputProps()} />
      <div className="text-3xl mb-3">{file ? '✓' : icon}</div>
      {file ? (
        <div>
          <div className="text-signal-green font-mono text-sm">{file.name}</div>
          <div className="text-white/30 text-xs mt-1">{(file.size / 1024).toFixed(1)} KB</div>
        </div>
      ) : (
        <div>
          <div className="text-white/70 text-sm font-medium mb-1">{label}</div>
          <div className="text-white/30 text-xs">{isDragActive ? 'Drop here' : 'Drag & drop or click to browse'}</div>
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

  const onDropModel = useCallback((files: File[]) => {
    if (files[0]) store.setModelFile(files[0])
  }, [])

  const onDropDataset = useCallback((files: File[]) => {
    if (files[0]) store.setDatasetFile(files[0])
  }, [])

  const addCol = () => {
    const c = newCol.trim()
    if (c && !store.protectedCols.includes(c)) store.setProtectedCols([...store.protectedCols, c])
    setNewCol('')
  }

  const canRun = store.datasetFile && store.protectedCols.length > 0 && store.targetCol

  const runAudit = async () => {
    if (!canRun) return
    store.setError(null)
    store.setLoading(true)
    try {
      setRunningStage('Bias Cartography'); setProgress(10)
      store.setStage('cartography')
      const carto = await runCartography(store.modelFile, store.datasetFile!, store.protectedCols, store.targetCol)
      store.setCartographyResults(carto); setProgress(40)

      setRunningStage('Counterfactual Constitution')
      store.setStage('constitution')
      const constitution = await runConstitution(store.modelFile, store.datasetFile!, store.protectedCols, store.targetCol, carto)
      store.setConstitutionResults(constitution); setProgress(70)

      setRunningStage('Proxy Variable Hunter')
      store.setStage('proxy')
      const proxy = await runProxyHunter(store.datasetFile!, store.protectedCols, store.targetCol)
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
          Upload your model and dataset. FairLens analyses bias using Gemini 1.5 Pro on Google Cloud.
        </p>
      </motion.div>

      {/* File uploads */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
        className="grid md:grid-cols-2 gap-4 mb-6">
        <DropZone
          label="Trained Model (.pkl) — optional"
          accept={{ 'application/octet-stream': ['.pkl'] }}
          file={store.modelFile}
          onDrop={onDropModel}
          icon="🧠"
        />
        <DropZone
          label="Dataset (.csv) — required"
          accept={{ 'text/csv': ['.csv'] }}
          file={store.datasetFile}
          onDrop={onDropDataset}
          icon="📊"
        />
      </motion.div>

      {/* Protected columns */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}
        className="glass rounded-2xl p-5 border border-white/5 mb-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-display font-semibold text-white text-sm">Protected Attribute Columns</h3>
          <button
            onClick={() => { store.setProtectedCols(['gender', 'race', 'age_group']); store.setTargetCol('hired') }}
            className="text-xs font-mono text-lens-light border border-lens/20 px-3 py-1 rounded-full hover:bg-lens/10 transition-all">
            Load demo
          </button>
        </div>
        <p className="text-white/25 text-xs mb-3">Columns containing demographic info: gender, race, age, nationality…</p>
        <div className="flex flex-wrap gap-2 mb-3 min-h-[28px]">
          {store.protectedCols.map(c => (
            <ColTag key={c} col={c} onRemove={() => store.setProtectedCols(store.protectedCols.filter(x => x !== c))} />
          ))}
          {!store.protectedCols.length && <span className="text-white/20 text-xs font-mono">None added yet</span>}
        </div>
        <div className="flex gap-2">
          <input value={newCol} onChange={e => setNewCol(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && addCol()}
            placeholder="e.g. gender"
            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono"
          />
          <button onClick={addCol}
            className="bg-lens/20 hover:bg-lens/30 border border-lens/30 text-lens-light px-4 rounded-xl text-sm transition-all">
            Add
          </button>
        </div>
      </motion.div>

      {/* Target column */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
        className="glass rounded-2xl p-5 border border-white/5 mb-8">
        <h3 className="font-display font-semibold text-white text-sm mb-2">Target Column</h3>
        <p className="text-white/25 text-xs mb-3">The column being predicted — e.g. hired, approved, risk_score</p>
        <input value={store.targetCol} onChange={e => store.setTargetCol(e.target.value)}
          placeholder="e.g. hired"
          className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder-white/20 focus:outline-none focus:border-signal-green/50 font-mono"
        />
      </motion.div>

      <AnimatePresence>
        {store.error && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="bg-signal-red/10 border border-signal-red/30 rounded-xl p-4 mb-6 text-signal-red text-sm font-mono">
            ⚠ {store.error}
          </motion.div>
        )}
      </AnimatePresence>

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
        onClick={runAudit}
        className={`w-full py-4 rounded-xl font-display font-semibold text-base transition-all
          ${canRun && !store.loading
            ? 'bg-lens hover:bg-lens/90 text-white glow-lens cursor-pointer'
            : 'bg-white/5 text-white/20 cursor-not-allowed'}`}>
        {store.loading ? 'Analysing...' : 'Run Full Bias Audit →'}
      </motion.button>

      <p className="text-center text-white/20 text-xs mt-3 font-mono">
        Powered by Gemini 1.5 Pro · Google Cloud · Model file optional
      </p>
    </div>
  )
}
