import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuditStore } from '../hooks/useAuditStore'
import { streamRedTeam } from '../utils/api'

type AgentEvent = {
  node: string
  iteration?: number
  log?: string[]
  status?: string
  results?: any
}

const NODE_META: Record<string, { label: string; icon: string; color: string }> = {
  attack:       { label: 'Attack Agent',     icon: '⚔',  color: '#ef4444' },
  evaluate:     { label: 'Evaluator Agent',  icon: '🔬', color: '#f97316' },
  decide_patch: { label: 'Decision Agent',   icon: '⚖',  color: '#eab308' },
  patch:        { label: 'Patcher Agent',    icon: '🔧', color: '#3b82f6' },
  validate:     { label: 'Validator Agent',  icon: '✓',  color: '#10b981' },
  report:       { label: 'Report Agent',     icon: '📋', color: '#7c3aed' },
  complete:     { label: 'Complete',         icon: '✅', color: '#10b981' },
}

function AgentNode({ name, active, done }: { name: string; active: boolean; done: boolean }) {
  const meta = NODE_META[name] || { label: name, icon: '◈', color: '#7c3aed' }
  return (
    <motion.div
      animate={{ scale: active ? 1.05 : 1 }}
      className={`flex items-center gap-3 p-3 rounded-xl border transition-all ${
        active ? 'border-opacity-40 bg-opacity-10' : done ? 'border-white/10' : 'border-white/5'
      }`}
      style={{
        borderColor: active ? meta.color + '60' : undefined,
        background: active ? meta.color + '10' : undefined,
      }}
    >
      <div
        className="w-8 h-8 rounded-lg flex items-center justify-center text-sm flex-shrink-0"
        style={{ background: meta.color + '20', border: `1px solid ${meta.color}40` }}
      >
        {active ? (
          <span className="animate-spin text-xs inline-block">⟳</span>
        ) : done ? '✓' : meta.icon}
      </div>
      <div>
        <div className="text-xs font-mono" style={{ color: active ? meta.color : done ? '#10b981' : '#ffffff44' }}>
          {meta.label}
        </div>
        {active && (
          <div className="flex gap-0.5 mt-1">
            {[0, 1, 2].map(i => (
              <motion.div key={i} className="w-1 h-1 rounded-full"
                style={{ background: meta.color }}
                animate={{ opacity: [0.3, 1, 0.3] }}
                transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.2 }}
              />
            ))}
          </div>
        )}
      </div>
    </motion.div>
  )
}

function LogLine({ line, index }: { line: string; index: number }) {
  const getColor = (text: string) => {
    if (text.includes('Attack Agent')) return '#ef4444'
    if (text.includes('Evaluator')) return '#f97316'
    if (text.includes('Decision')) return '#eab308'
    if (text.includes('Patcher')) return '#3b82f6'
    if (text.includes('Validator')) return '#10b981'
    if (text.includes('Report')) return '#7c3aed'
    if (text.includes('Failed') || text.includes('error')) return '#ef4444'
    if (text.includes('improved') || text.includes('Done')) return '#10b981'
    return '#ffffff50'
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      className="flex items-start gap-2 py-1 font-mono text-xs border-b border-white/3 last:border-0"
    >
      <span className="text-white/20 flex-shrink-0 mt-0.5">{String(index + 1).padStart(3, '0')}</span>
      <span style={{ color: getColor(line) }}>{line}</span>
    </motion.div>
  )
}

function MetricCard({ label, before, after, unit = '' }: any) {
  const improved = after < before
  return (
    <div className="glass rounded-xl p-4 border border-white/5">
      <div className="text-white/30 text-xs font-mono mb-2">{label}</div>
      <div className="flex items-center gap-3">
        <div className="text-center">
          <div className="text-signal-red font-display font-bold text-lg">{(before * 100).toFixed(1)}{unit}</div>
          <div className="text-white/20 text-xs">before</div>
        </div>
        <div className="text-white/30 text-lg">→</div>
        <div className="text-center">
          <div className={`font-display font-bold text-lg ${improved ? 'text-signal-green' : 'text-signal-red'}`}>
            {(after * 100).toFixed(1)}{unit}
          </div>
          <div className="text-white/20 text-xs">after</div>
        </div>
        <div className={`text-xs font-mono px-2 py-0.5 rounded-full ml-auto ${
          improved ? 'bg-signal-green/10 text-signal-green border border-signal-green/20' : 'bg-signal-red/10 text-signal-red border border-signal-red/20'
        }`}>
          {improved ? '↓ improved' : '↑ regressed'}
        </div>
      </div>
    </div>
  )
}

export default function RedTeamPage() {
  const nav = useNavigate()
  const store = useAuditStore()
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [allLogs, setAllLogs] = useState<string[]>([])
  const [activeNode, setActiveNode] = useState<string | null>(null)
  const [doneNodes, setDoneNodes] = useState<Set<string>>(new Set())
  const [finalResults, setFinalResults] = useState<any>(null)
  const [running, setRunning] = useState(false)
  const [started, setStarted] = useState(false)
  const stopRef = useRef<(() => void) | null>(null)
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [allLogs])

  const startRedTeam = () => {
    if (!store.datasetFile || store.confirmedBiases.length === 0) return
    setRunning(true)
    setStarted(true)
    setEvents([])
    setAllLogs([])
    setDoneNodes(new Set())
    setFinalResults(null)

    const auditResults = {
      cartography: store.cartographyResults,
      constitution: store.constitutionResults,
      proxy: store.proxyResults,
    }

    const stop = streamRedTeam(
      store.datasetFile,
      store.protectedCols,
      store.targetCol,
      store.confirmedBiases,
      auditResults,
      (event: AgentEvent) => {
        setEvents(prev => [...prev, event])

        if (event.log?.length) {
          setAllLogs(prev => [...prev, ...event.log!])
        }

        if (event.node && event.node !== 'complete') {
          setActiveNode(event.node)
          setDoneNodes(prev => {
            const n = new Set(prev)
            // mark previous node done
            return n
          })
        }

        if (event.status === 'done' || event.node === 'complete') {
          setActiveNode(null)
          setDoneNodes(new Set(Object.keys(NODE_META)))
          setRunning(false)
          setFinalResults(event.results)
          store.setRedteamResults(event.results)
          store.setStage('done')
        }
      }
    )
    stopRef.current = stop
  }

  const stopRedTeam = () => {
    stopRef.current?.()
    setRunning(false)
    setActiveNode(null)
  }

  const AGENT_NODES = ['attack', 'evaluate', 'decide_patch', 'patch', 'validate', 'report']

  const validation = finalResults?.validation_results || {}
  const improved = validation.improved || []
  const patches = finalResults?.patches_applied || 0

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">

      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <div className="text-xs font-mono text-signal-red mb-1">Stage 6 — Red-Team Agent</div>
        <h1 className="font-display font-bold text-white text-3xl mb-2">Adversarial Bias Attack & Fix</h1>
        <p className="text-white/40 text-sm max-w-2xl">
          The agent will iteratively generate adversarial probes targeting confirmed biases,
          evaluate their impact, and apply mitigation patches — then validate that fixes hold.
        </p>
      </motion.div>

      {/* Confirmed biases summary */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        className="glass rounded-2xl p-5 border border-white/5 mb-6">
        <div className="text-xs font-mono text-white/30 mb-3 uppercase tracking-wider">Targeting {store.confirmedBiases.length} confirmed biases</div>
        <div className="flex flex-wrap gap-2">
          {store.confirmedBiases.map((b: any, i: number) => (
            <div key={i} className="flex items-center gap-2 bg-signal-red/10 border border-signal-red/20 rounded-lg px-3 py-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-signal-red" />
              <span className="text-signal-red text-xs font-mono">{b.attribute}</span>
            </div>
          ))}
        </div>
      </motion.div>

      <div className="grid md:grid-cols-3 gap-6">

        {/* Agent pipeline */}
        <div className="glass rounded-2xl p-5 border border-white/5">
          <h3 className="font-display font-semibold text-white text-sm mb-4">Agent Pipeline</h3>
          <div className="space-y-2">
            {AGENT_NODES.map(node => (
              <AgentNode
                key={node}
                name={node}
                active={activeNode === node}
                done={doneNodes.has(node)}
              />
            ))}
          </div>

          {!started && (
            <motion.button
              whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }}
              onClick={startRedTeam}
              className="w-full mt-6 py-3 rounded-xl bg-signal-red hover:bg-signal-red/90 text-white font-display font-semibold text-sm glow-red transition-all"
            >
              ⊘ Launch Red-Team Agent
            </motion.button>
          )}

          {running && (
            <button onClick={stopRedTeam}
              className="w-full mt-4 py-2.5 rounded-xl border border-white/10 text-white/40 hover:text-white/70 text-sm font-mono transition-all">
              Stop agent
            </button>
          )}

          {finalResults && (
            <div className="mt-4 p-3 rounded-xl bg-signal-green/5 border border-signal-green/20">
              <div className="text-signal-green font-mono text-xs font-semibold mb-1">✓ Agent complete</div>
              <div className="text-white/40 text-xs">
                {patches} patches applied · {store.confirmedBiases.length} biases targeted
              </div>
            </div>
          )}
        </div>

        {/* Live log */}
        <div className="md:col-span-2 glass rounded-2xl border border-white/5 overflow-hidden flex flex-col" style={{ maxHeight: '460px' }}>
          <div className="flex items-center justify-between px-5 py-3 border-b border-white/5">
            <h3 className="font-display font-semibold text-white text-sm">Agent Activity Log</h3>
            {running && (
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-signal-red animate-pulse" />
                <span className="text-signal-red text-xs font-mono">LIVE</span>
              </div>
            )}
          </div>

          <div ref={logRef} className="flex-1 overflow-y-auto px-5 py-4 min-h-[200px]">
            {allLogs.length === 0 && !started && (
              <div className="text-center py-12">
                <div className="text-4xl mb-4">⊘</div>
                <div className="text-white/30 text-sm font-mono">
                  Launch the agent to start the red-team attack
                </div>
              </div>
            )}
            {allLogs.map((line, i) => (
              <LogLine key={i} line={line} index={i} />
            ))}
            {running && (
              <div className="flex items-center gap-2 mt-3 text-xs font-mono text-white/30">
                <motion.span animate={{ opacity: [0.3, 1, 0.3] }} transition={{ duration: 1, repeat: Infinity }}>
                  ▋
                </motion.span>
                Agent running...
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Results panel — shown after completion */}
      <AnimatePresence>
        {finalResults && (
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-8"
          >
            <h2 className="font-display font-bold text-white text-2xl mb-6">
              Red-Team Report
            </h2>

            {/* Metrics grid */}
            <div className="grid md:grid-cols-3 gap-4 mb-6">
              <div className="glass rounded-2xl p-5 border border-white/5">
                <div className="text-white/30 text-xs font-mono mb-2">Agent Iterations</div>
                <div className="font-display font-bold text-3xl text-white">{finalResults.iterations || 1}</div>
              </div>
              <div className="glass rounded-2xl p-5 border border-white/5">
                <div className="text-white/30 text-xs font-mono mb-2">Patches Applied</div>
                <div className="font-display font-bold text-3xl text-signal-green">{patches}</div>
              </div>
              <div className="glass rounded-2xl p-5 border border-white/5">
                <div className="text-white/30 text-xs font-mono mb-2">Biases Improved</div>
                <div className="font-display font-bold text-3xl text-signal-green">{improved.length}</div>
              </div>
            </div>

            {/* Improvement cards */}
            {improved.length > 0 && (
              <div className="mb-6">
                <h3 className="font-display font-semibold text-white text-sm mb-3">Bias Disparity — Before vs After</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  {improved.filter((imp: any) => typeof imp === 'object').map((imp: any, i: number) => (
                    <MetricCard
                      key={i}
                      label={`Attribute: ${imp.attribute}`}
                      before={imp.before}
                      after={imp.after}
                      unit="%"
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Log tail */}
            {finalResults.log_summary?.length > 0 && (
              <div className="glass rounded-2xl p-5 border border-white/5 mb-6">
                <h3 className="font-display font-semibold text-white text-sm mb-3">Final Agent Notes</h3>
                {finalResults.log_summary.map((line: string, i: number) => (
                  <div key={i} className="text-white/40 text-xs font-mono py-1 border-b border-white/5 last:border-0">{line}</div>
                ))}
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-4">
              <button
                onClick={() => nav('/audit')}
                className="flex-1 py-3 rounded-xl border border-white/10 text-white/60 hover:text-white font-mono text-sm transition-all"
              >
                ← New Audit
              </button>
              <button
                onClick={() => {
                  const blob = new Blob([JSON.stringify({
                    cartography: store.cartographyResults,
                    constitution: store.constitutionResults,
                    proxy: store.proxyResults,
                    redteam: finalResults,
                  }, null, 2)], { type: 'application/json' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `fairlens-report-${store.cartographyResults?.audit_id || 'export'}.json`
                  a.click()
                  URL.revokeObjectURL(url)
                }}
                className="flex-1 py-3 rounded-xl bg-lens hover:bg-lens/90 text-white font-display font-semibold text-sm glow-lens transition-all"
              >
                ↓ Export Full Report
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
