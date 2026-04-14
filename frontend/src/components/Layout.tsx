import { Outlet, useLocation, Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useAuditStore } from '../hooks/useAuditStore'

const STAGES = [
  { key: 'upload',       label: 'Upload',        path: '/audit',   num: 1 },
  { key: 'cartography',  label: 'Cartography',   path: '/results', num: 2 },
  { key: 'constitution', label: 'Constitution',  path: '/results', num: 3 },
  { key: 'proxy',        label: 'Proxy Hunt',    path: '/results', num: 4 },
  { key: 'review',       label: 'Review',        path: '/results', num: 5 },
  { key: 'redteam',      label: 'Red-Team',      path: '/redteam', num: 6 },
]

const STAGE_ORDER = STAGES.map(s => s.key)

export default function Layout() {
  const { stage } = useAuditStore()
  const loc = useLocation()
  const currentIdx = STAGE_ORDER.indexOf(stage)

  return (
    <div className="min-h-screen bg-night flex flex-col">
      {/* Top nav */}
      <header className="border-b border-white/5 sticky top-0 z-50 bg-night/90 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
          <Link to="/" className="font-display font-bold text-lg text-white flex items-center gap-2">
            <span className="text-lens">◈</span> FairLens
          </Link>

          {/* Stage progress */}
          <div className="hidden md:flex items-center gap-1">
            {STAGES.map((s, i) => {
              const done = i < currentIdx
              const active = i === currentIdx
              return (
                <div key={s.key} className="flex items-center">
                  <div className={`
                    flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono transition-all
                    ${active ? 'bg-lens/20 text-lens-light border border-lens/30' : ''}
                    ${done ? 'text-signal-green' : ''}
                    ${!active && !done ? 'text-white/20' : ''}
                  `}>
                    {done ? '✓' : s.num}
                    <span className={active ? 'text-lens-light' : ''}>{s.label}</span>
                  </div>
                  {i < STAGES.length - 1 && (
                    <div className={`w-4 h-px mx-0.5 ${i < currentIdx ? 'bg-signal-green/40' : 'bg-white/10'}`} />
                  )}
                </div>
              )
            })}
          </div>

          <div className="text-xs font-mono text-white/30">
            Google Solution Challenge 2026
          </div>
        </div>
      </header>

      <main className="flex-1">
        <motion.div
          key={loc.pathname}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Outlet />
        </motion.div>
      </main>
    </div>
  )
}
