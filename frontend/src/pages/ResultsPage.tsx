import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import * as d3 from 'd3'
import ReactMarkdown from 'react-markdown'
import { useAuditStore } from '../hooks/useAuditStore'

// ── Bias Map (D3 scatter) ─────────────────────────────────────────────────
function BiasMap({ points, hotspots }: { points: any[]; hotspots: any[] }) {
  const ref = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!ref.current || !points.length) return
    const svg = d3.select(ref.current)
    svg.selectAll('*').remove()

    const W = ref.current.clientWidth || 480
    const H = 320
    const margin = { top: 16, right: 16, bottom: 32, left: 32 }
    const w = W - margin.left - margin.right
    const h = H - margin.top - margin.bottom

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    const xExt = d3.extent(points, (d: any) => d.x) as [number, number]
    const yExt = d3.extent(points, (d: any) => d.y) as [number, number]
    const xScale = d3.scaleLinear().domain(xExt).range([0, w])
    const yScale = d3.scaleLinear().domain(yExt).range([h, 0])
    const colorScale = d3.scaleSequential(d3.interpolateRdYlGn).domain([1, 0])

    // Points
    g.selectAll('circle.pt')
      .data(points.slice(0, 1500))
      .join('circle')
      .attr('class', 'pt')
      .attr('cx', (d: any) => xScale(d.x))
      .attr('cy', (d: any) => yScale(d.y))
      .attr('r', 2.5)
      .attr('fill', (d: any) => colorScale(d.bias_score))
      .attr('opacity', 0.7)

    // Hotspot circles
    hotspots.forEach((h: any) => {
      g.append('circle')
        .attr('cx', xScale(h.centroid_x))
        .attr('cy', yScale(h.centroid_y))
        .attr('r', 18)
        .attr('fill', 'none')
        .attr('stroke', '#ef4444')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4 2')
        .attr('opacity', 0.8)

      g.append('text')
        .attr('x', xScale(h.centroid_x))
        .attr('y', yScale(h.centroid_y) - 22)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ef4444')
        .attr('font-size', 9)
        .attr('font-family', 'JetBrains Mono, monospace')
        .text(h.dominant_slice)
    })

    // Legend
    const defs = svg.append('defs')
    const grad = defs.append('linearGradient').attr('id', 'bias-grad').attr('x1', '0%').attr('x2', '100%')
    grad.append('stop').attr('offset', '0%').attr('stop-color', '#ef4444')
    grad.append('stop').attr('offset', '50%').attr('stop-color', '#eab308')
    grad.append('stop').attr('offset', '100%').attr('stop-color', '#22c55e')

    svg.append('rect')
      .attr('x', W - 110).attr('y', H - 16).attr('width', 90).attr('height', 6)
      .attr('fill', 'url(#bias-grad)').attr('rx', 3)
    svg.append('text').attr('x', W - 115).attr('y', H - 20).attr('fill', '#ffffff44')
      .attr('font-size', 8).attr('font-family', 'JetBrains Mono, monospace').text('high bias')
    svg.append('text').attr('x', W - 30).attr('y', H - 20).attr('fill', '#ffffff44')
      .attr('font-size', 8).attr('font-family', 'JetBrains Mono, monospace').text('low')

  }, [points, hotspots])

  return (
    <svg ref={ref} width="100%" height={320}
      className="rounded-xl overflow-visible" style={{ background: 'rgba(255,255,255,0.02)' }} />
  )
}

// ── Proxy Graph (D3 force) ────────────────────────────────────────────────
function ProxyGraph({ graph, chains }: { graph: any; chains: any[] }) {
  const ref = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!ref.current || !graph?.nodes?.length) return
    const svg = d3.select(ref.current)
    svg.selectAll('*').remove()

    const W = ref.current.clientWidth || 480
    const H = 280

    const riskMap: Record<string, string> = {}
    chains.forEach((c: any) => { riskMap[c.start_feature] = c.risk_level })

    const nodeColor = (n: any) => {
      if (n.is_protected) return '#7c3aed'
      const r = riskMap[n.id]
      return r === 'critical' ? '#ef4444' : r === 'high' ? '#f97316' : r === 'medium' ? '#eab308' : '#334155'
    }

    const sim = d3.forceSimulation(graph.nodes)
      .force('link', d3.forceLink(graph.edges).id((d: any) => d.id).distance(60))
      .force('charge', d3.forceManyBody().strength(-80))
      .force('center', d3.forceCenter(W / 2, H / 2))

    const link = svg.append('g').selectAll('line')
      .data(graph.edges).join('line')
      .attr('stroke', (d: any) => `rgba(124,58,237,${Math.min(d.weight * 1.5, 0.6)})`)
      .attr('stroke-width', (d: any) => Math.max(d.weight * 3, 0.5))

    const node = svg.append('g').selectAll('g')
      .data(graph.nodes).join('g').attr('cursor', 'pointer')

    node.append('circle').attr('r', (d: any) => d.is_protected ? 9 : 6)
      .attr('fill', nodeColor).attr('fill-opacity', 0.85)

    node.append('text')
      .attr('dy', -11).attr('text-anchor', 'middle')
      .attr('fill', '#ffffff88').attr('font-size', 7.5)
      .attr('font-family', 'JetBrains Mono, monospace')
      .text((d: any) => d.id.length > 12 ? d.id.slice(0, 12) + '…' : d.id)

    sim.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x).attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x).attr('y2', (d: any) => d.target.y)
      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`)
    })
  }, [graph, chains])

  return (
    <svg ref={ref} width="100%" height={280}
      className="rounded-xl" style={{ background: 'rgba(255,255,255,0.02)' }} />
  )
}

// ── Severity badge ────────────────────────────────────────────────────────
function SeverityBadge({ level }: { level: string }) {
  const map: Record<string, string> = {
    critical: 'bg-bias-critical text-red-300 border border-red-500/30',
    high: 'bg-bias-high text-orange-300 border border-orange-500/30',
    medium: 'bg-bias-medium text-yellow-300 border border-yellow-500/30',
    low: 'bg-bias-low text-green-300 border border-green-500/30',
  }
  return (
    <span className={`text-xs font-mono px-2 py-0.5 rounded ${map[level] || map.low}`}>
      {level}
    </span>
  )
}

// ── Main Results Page ─────────────────────────────────────────────────────
export default function ResultsPage() {
  const nav = useNavigate()
  const store = useAuditStore()
  const [activeTab, setActiveTab] = useState<'cartography' | 'constitution' | 'proxy'>('cartography')
  const [selectedBiases, setSelectedBiases] = useState<Set<string>>(new Set())

  const carto = store.cartographyResults
  const constitution = store.constitutionResults
  const proxy = store.proxyResults

  if (!carto) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-24 text-center">
        <div className="text-white/30 text-sm font-mono mb-4">No audit results found.</div>
        <button onClick={() => nav('/audit')} className="text-lens-light hover:underline text-sm">← Go back to audit</button>
      </div>
    )
  }

  const toggleBias = (key: string) => {
    const next = new Set(selectedBiases)
    next.has(key) ? next.delete(key) : next.add(key)
    setSelectedBiases(next)
  }

  const confirmedBiases = [
    ...(carto?.slice_metrics?.filter((m: any) => m.flagged).map((m: any) => ({
      attribute: m.attributes ? Object.keys(m.attributes)[0] : 'unknown',
      type: 'demographic_parity',
      label: m.label,
      spd: m.statistical_parity_diff,
    })) || []),
    ...(proxy?.proxy_chains?.filter((c: any) => c.risk_level === 'critical' || c.risk_level === 'high').map((c: any) => ({
      attribute: c.start_feature,
      type: 'proxy',
      label: c.explanation,
      chain: c.path,
    })) || []),
  ]

  const selectedList = confirmedBiases.filter((b: any) => selectedBiases.has(b.label || b.attribute))

  const proceedToRedTeam = () => {
    store.setConfirmedBiases(selectedList)
    store.setStage('redteam')
    nav('/redteam')
  }

  const TABS = [
    { key: 'cartography', label: 'Bias Map', count: carto?.hotspots?.length },
    { key: 'constitution', label: 'Constitution', count: constitution?.summary?.decision_flips },
    { key: 'proxy', label: 'Proxy Chains', count: proxy?.summary?.critical_proxies },
  ]

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">

      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xs font-mono text-lens-light mb-1">Audit Complete — Review Findings</div>
            <h1 className="font-display font-bold text-white text-3xl">Bias Analysis Report</h1>
          </div>
          <div className="text-xs font-mono text-white/30 glass rounded-xl px-4 py-2 border border-white/5">
            Audit ID: {carto?.audit_id}
          </div>
        </div>
      </motion.div>

      {/* Summary cards */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
        {[
          { label: 'Samples Analysed', value: carto?.summary?.total_samples?.toLocaleString(), color: 'text-white' },
          { label: 'Bias Hotspots', value: carto?.summary?.hotspot_count, color: 'text-signal-red' },
          { label: 'Decision Flips', value: constitution?.summary?.decision_flips, color: 'text-signal-amber' },
          { label: 'Critical Proxies', value: proxy?.summary?.critical_proxies, color: 'text-lens-light' },
        ].map((card, i) => (
          <div key={i} className="glass rounded-2xl p-4 border border-white/5">
            <div className="text-white/30 text-xs font-mono mb-1">{card.label}</div>
            <div className={`font-display font-bold text-2xl ${card.color}`}>{card.value ?? '—'}</div>
          </div>
        ))}
      </motion.div>

      {/* Tab navigation */}
      <div className="flex gap-2 mb-6">
        {TABS.map(tab => (
          <button key={tab.key} onClick={() => setActiveTab(tab.key as any)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-mono transition-all
              ${activeTab === tab.key ? 'bg-lens/20 text-lens-light border border-lens/30' : 'text-white/40 hover:text-white/70'}`}>
            {tab.label}
            {tab.count != null && (
              <span className={`text-xs px-1.5 py-0.5 rounded-full ${activeTab === tab.key ? 'bg-lens/30' : 'bg-white/5'}`}>
                {tab.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab panels */}
      <AnimatePresence mode="wait">
        <motion.div key={activeTab} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }}
          className="mb-10">

          {activeTab === 'cartography' && (
            <div className="grid md:grid-cols-3 gap-6">
              <div className="md:col-span-2 glass rounded-2xl p-5 border border-white/5">
                <h3 className="font-display font-semibold text-white text-sm mb-1">Bias Topology Map</h3>
                <p className="text-white/30 text-xs mb-4 font-mono">
                  Each point is a model prediction. Colour = bias magnitude. Red circles = hotspot clusters.
                </p>
                <BiasMap points={carto?.map_points || []} hotspots={carto?.hotspots || []} />
              </div>
              <div className="glass rounded-2xl p-5 border border-white/5 overflow-y-auto max-h-[400px]">
                <h3 className="font-display font-semibold text-white text-sm mb-4">Hotspots</h3>
                {carto?.hotspots?.map((h: any, i: number) => (
                  <div key={i} className="border-b border-white/5 pb-3 mb-3 last:border-0">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-white/70 text-xs font-mono">{h.dominant_slice}</span>
                      <SeverityBadge level={h.severity} />
                    </div>
                    <div className="text-white/30 text-xs">
                      {h.size} samples · bias score {h.mean_bias_magnitude?.toFixed(3)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'constitution' && constitution && (
            <div className="glass rounded-2xl p-6 border border-white/5">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-8 h-8 rounded-xl bg-lens/20 border border-lens/30 flex items-center justify-center text-sm">◍</div>
                <div>
                  <h3 className="font-display font-semibold text-white text-sm">Counterfactual Constitution</h3>
                  <div className="text-white/30 text-xs font-mono">
                    Flip rate: {(constitution.summary?.flip_rate * 100).toFixed(1)}% · Most sensitive: {constitution.summary?.most_sensitive_attribute}
                  </div>
                </div>
              </div>
              <div className="prose prose-invert prose-sm max-w-none text-white/70 prose-headings:font-display prose-headings:text-white prose-headings:font-semibold prose-code:font-mono prose-code:text-lens-light">
                <ReactMarkdown>{constitution.constitution_markdown || '_No constitution generated._'}</ReactMarkdown>
              </div>
            </div>
          )}

          {activeTab === 'proxy' && proxy && (
            <div className="grid md:grid-cols-2 gap-6">
              <div className="glass rounded-2xl p-5 border border-white/5">
                <h3 className="font-display font-semibold text-white text-sm mb-1">Proxy Dependency Graph</h3>
                <p className="text-white/30 text-xs mb-4 font-mono">
                  Purple = protected attributes. Red/orange = high-risk proxies.
                </p>
                <ProxyGraph graph={proxy.graph} chains={proxy.proxy_chains || []} />
              </div>
              <div className="glass rounded-2xl p-5 border border-white/5 overflow-y-auto max-h-[400px]">
                <h3 className="font-display font-semibold text-white text-sm mb-4">Proxy Chains</h3>
                {proxy.proxy_chains?.slice(0, 15).map((c: any, i: number) => (
                  <div key={i} className="border-b border-white/5 pb-3 mb-3 last:border-0">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-white/70 text-xs font-mono">{c.path?.join(' → ')}</span>
                      <SeverityBadge level={c.risk_level} />
                    </div>
                    <div className="text-white/30 text-xs">{c.explanation}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Confirm biases section */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
        className="glass rounded-2xl p-6 border border-signal-red/10">
        <div className="flex items-center gap-3 mb-2">
          <span className="text-signal-red">⊘</span>
          <h3 className="font-display font-semibold text-white text-base">Confirm Biases for Red-Team Agent</h3>
        </div>
        <p className="text-white/40 text-sm mb-6">
          Select which bias findings you want the Red-Team Agent to attack and fix.
          The agent will only target issues you confirm.
        </p>

        <div className="space-y-2 mb-6">
          {confirmedBiases.map((b: any, i: number) => {
            const key = b.label || b.attribute
            const checked = selectedBiases.has(key)
            return (
              <button key={i} onClick={() => toggleBias(key)}
                className={`w-full text-left flex items-start gap-3 p-4 rounded-xl border transition-all
                  ${checked ? 'border-signal-red/30 bg-signal-red/5' : 'border-white/5 bg-white/2 hover:border-white/15'}`}>
                <div className={`w-4 h-4 rounded mt-0.5 border flex items-center justify-center flex-shrink-0 transition-all
                  ${checked ? 'bg-signal-red border-signal-red' : 'border-white/20'}`}>
                  {checked && <span className="text-white text-xs">✓</span>}
                </div>
                <div>
                  <div className="text-white/80 text-xs font-mono mb-1 flex items-center gap-2">
                    <span className="font-semibold">{b.attribute}</span>
                    <SeverityBadge level={b.type === 'proxy' ? 'high' : 'critical'} />
                  </div>
                  <div className="text-white/40 text-xs">{b.label || b.explanation}</div>
                </div>
              </button>
            )
          })}
          {confirmedBiases.length === 0 && (
            <div className="text-white/20 text-sm font-mono text-center py-6">
              No flagged biases found — your model looks fair!
            </div>
          )}
        </div>

        <button
          onClick={proceedToRedTeam}
          disabled={selectedBiases.size === 0}
          className={`w-full py-4 rounded-xl font-display font-semibold transition-all
            ${selectedBiases.size > 0
              ? 'bg-signal-red hover:bg-signal-red/90 text-white glow-red cursor-pointer'
              : 'bg-white/5 text-white/20 cursor-not-allowed'}`}>
          Launch Red-Team Agent on {selectedBiases.size} confirmed bias{selectedBiases.size !== 1 ? 'es' : ''} →
        </button>
      </motion.div>
    </div>
  )
}
