import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import * as d3 from 'd3'
import ReactMarkdown from 'react-markdown'
import { useAuditStore } from '../hooks/useAuditStore'
import { exportPdfReport } from '../utils/api'

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

// ── CI Error Bar Chart (D3) ───────────────────────────────────────────────
function CIChart({ ciData }: { ciData: Record<string, Record<string, any>> }) {
  const ref = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!ref.current) return
    const allGroups: { col: string; val: string; mean: number; lower: number; upper: number }[] = []
    Object.entries(ciData).forEach(([col, vals]) => {
      Object.entries(vals).forEach(([val, stats]: [string, any]) => {
        allGroups.push({ col, val, mean: stats.spd_mean, lower: stats.spd_lower, upper: stats.spd_upper })
      })
    })
    if (!allGroups.length) return

    const svg = d3.select(ref.current)
    svg.selectAll('*').remove()

    const W = ref.current.clientWidth || 460
    const H = 220
    const margin = { top: 16, right: 20, bottom: 48, left: 52 }
    const w = W - margin.left - margin.right
    const h = H - margin.top - margin.bottom

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    const labels = allGroups.map(d => d.val)
    const xScale = d3.scaleBand().domain(labels).range([0, w]).padding(0.4)
    const yExtent = [
      Math.min(-0.01, d3.min(allGroups, (d: any) => d.lower) as number) * 1.2,
      Math.max(0.01,  d3.max(allGroups, (d: any) => d.upper) as number) * 1.2,
    ]
    const yScale = d3.scaleLinear().domain(yExtent).range([h, 0])

    // Zero line
    g.append('line').attr('x1', 0).attr('x2', w).attr('y1', yScale(0)).attr('y2', yScale(0))
      .attr('stroke', 'rgba(255,255,255,0.15)').attr('stroke-dasharray', '4 3')

    // Axes
    g.append('g').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).tickSize(0))
      .selectAll('text').attr('fill', '#ffffff55').attr('font-size', 9).attr('font-family', 'JetBrains Mono, monospace')
      .attr('transform', 'rotate(-30)').attr('text-anchor', 'end').attr('dy', '0.8em')
    g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat((d: any) => `${(+d * 100).toFixed(0)}%`))
      .selectAll('text').attr('fill', '#ffffff55').attr('font-size', 8).attr('font-family', 'JetBrains Mono, monospace')
    g.select('.domain').remove()

    // Bars
    g.selectAll('rect.bar')
      .data(allGroups).join('rect').attr('class', 'bar')
      .attr('x', d => xScale(d.val)!)
      .attr('width', xScale.bandwidth())
      .attr('y', d => d.mean >= 0 ? yScale(d.mean) : yScale(0))
      .attr('height', d => Math.abs(yScale(d.mean) - yScale(0)))
      .attr('fill', d => Math.abs(d.mean) > 0.1 ? '#ef4444' : Math.abs(d.mean) > 0.05 ? '#eab308' : '#22c55e')
      .attr('opacity', 0.7).attr('rx', 2)

    // Error whiskers
    const whiskerX = (d: typeof allGroups[0]) => (xScale(d.val) ?? 0) + xScale.bandwidth() / 2
    g.selectAll('line.whisker-v')
      .data(allGroups).join('line').attr('class', 'whisker-v')
      .attr('x1', whiskerX).attr('x2', whiskerX)
      .attr('y1', d => yScale(d.upper)).attr('y2', d => yScale(d.lower))
      .attr('stroke', '#ffffff99').attr('stroke-width', 1.5)
    g.selectAll('line.whisker-top')
      .data(allGroups).join('line').attr('class', 'whisker-top')
      .attr('x1', d => whiskerX(d) - 4).attr('x2', d => whiskerX(d) + 4)
      .attr('y1', d => yScale(d.upper)).attr('y2', d => yScale(d.upper))
      .attr('stroke', '#ffffff99').attr('stroke-width', 1.5)
    g.selectAll('line.whisker-bot')
      .data(allGroups).join('line').attr('class', 'whisker-bot')
      .attr('x1', d => whiskerX(d) - 4).attr('x2', d => whiskerX(d) + 4)
      .attr('y1', d => yScale(d.lower)).attr('y2', d => yScale(d.lower))
      .attr('stroke', '#ffffff99').attr('stroke-width', 1.5)

    // Y label
    svg.append('text').attr('transform', 'rotate(-90)')
      .attr('x', -(H / 2)).attr('y', 13)
      .attr('text-anchor', 'middle').attr('fill', '#ffffff33')
      .attr('font-size', 8).attr('font-family', 'JetBrains Mono, monospace')
      .text('Statistical Parity Diff (95% CI)')
  }, [ciData])

  return (
    <svg ref={ref} width="100%" height={220}
      className="rounded-xl" style={{ background: 'rgba(255,255,255,0.02)' }} />
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

// ── FairScore Gauge ───────────────────────────────────────────────────────
function FairScoreGauge({ score, label, color }: { score: number; label: string; color: string }) {
  const [displayed, setDisplayed] = useState(0)

  useEffect(() => {
    let start = 0
    const step = Math.ceil(score / 40)
    const timer = setInterval(() => {
      start += step
      if (start >= score) { setDisplayed(score); clearInterval(timer) }
      else setDisplayed(start)
    }, 30)
    return () => clearInterval(timer)
  }, [score])

  const radius = 54
  const circumference = 2 * Math.PI * radius
  const dashOffset = circumference - (displayed / 100) * circumference

  const strokeColor = color === 'green' ? '#22c55e' : color === 'yellow' ? '#eab308' : '#ef4444'
  const glowColor   = color === 'green' ? '#22c55e44' : color === 'yellow' ? '#eab30844' : '#ef444444'
  const textColor   = color === 'green' ? 'text-green-400' : color === 'yellow' ? 'text-yellow-400' : 'text-red-400'

  return (
    <div className="glass rounded-2xl p-6 border border-white/5 flex flex-col items-center justify-center">
      <div className="text-xs font-mono text-white/40 uppercase tracking-widest mb-4">FairScore™</div>
      <div className="relative" style={{ filter: `drop-shadow(0 0 12px ${glowColor})` }}>
        <svg width={140} height={140} viewBox="0 0 140 140">
          <circle cx={70} cy={70} r={radius} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth={10} />
          <circle
            cx={70} cy={70} r={radius} fill="none"
            stroke={strokeColor} strokeWidth={10}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            transform="rotate(-90 70 70)"
            style={{ transition: 'stroke-dashoffset 0.03s linear' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`font-display font-bold text-3xl ${textColor}`}>{displayed}</span>
          <span className="text-white/30 text-xs font-mono">/100</span>
        </div>
      </div>
      <div className={`font-mono text-sm font-semibold mt-3 ${textColor}`}>{label}</div>
      <div className="text-white/25 text-xs mt-1">
        {color === 'green' ? 'Model meets fairness standards' :
         color === 'yellow' ? 'Some bias patterns detected' :
         'Significant bias — action required'}
      </div>
    </div>
  )
}

// ── Compliance Badges ─────────────────────────────────────────────────────
function ComplianceBadges({ tags }: { tags: any[] }) {
  if (!tags?.length) return null
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}
      className="glass rounded-2xl p-5 border border-white/5 mb-6">
      <div className="text-xs font-mono text-white/40 uppercase tracking-widest mb-4">Regulatory Compliance</div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {tags.map((tag: any) => {
          const isPass = tag.status === 'PASS'
          const isFail = tag.status === 'FAIL'
          return (
            <div key={tag.id}
              className={`relative group rounded-xl p-3 border transition-all
                ${isPass
                  ? 'border-green-500/20 bg-green-500/5'
                  : isFail
                  ? 'border-red-500/20 bg-red-500/5'
                  : 'border-yellow-500/20 bg-yellow-500/5'}`}>
              <div className="flex items-center gap-1.5 mb-1">
                <span className={`text-sm ${isPass ? 'text-green-400' : isFail ? 'text-red-400' : 'text-yellow-400'}`}>
                  {isPass ? '✓' : isFail ? '✗' : '⚠'}
                </span>
                <span className={`text-xs font-mono font-semibold ${isPass ? 'text-green-300' : isFail ? 'text-red-300' : 'text-yellow-300'}`}>
                  {tag.status}
                </span>
              </div>
              <div className="text-white/70 text-xs font-mono leading-tight">{tag.label}</div>
              <div className="text-white/30 text-xs mt-0.5">{tag.domain}</div>
              {tag.violations?.length > 0 && (
                <div className="absolute bottom-full left-0 mb-2 w-64 hidden group-hover:block z-10
                  bg-gray-900 border border-white/10 rounded-xl p-3 shadow-xl text-xs text-white/60 font-mono leading-relaxed">
                  {tag.violations.map((v: string, i: number) => <div key={i}>{v}</div>)}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </motion.div>
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
  const [downloading, setDownloading] = useState(false)

  const downloadReport = async () => {
    if (!store.cartographyResults) return
    setDownloading(true)
    try {
      await exportPdfReport(store.cartographyResults)
    } catch (e: any) {
      alert(`PDF export failed: ${e.message}`)
    } finally {
      setDownloading(false)
    }
  }

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
      attribute: m.attribute || 'unknown',
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

  const highRiskProxies = (proxy?.summary?.critical_proxies ?? 0) + (proxy?.summary?.high_proxies ?? 0)

  const TABS = [
    { key: 'cartography', label: 'Bias Map', count: carto?.hotspots?.length },
    { key: 'constitution', label: 'Constitution', count: constitution?.summary?.decision_flips },
    { key: 'proxy', label: 'Proxy Chains', count: highRiskProxies },
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
          <div className="flex items-center gap-3">
            <button
              onClick={downloadReport}
              disabled={downloading}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-mono font-semibold border transition-all
                ${downloading
                  ? 'border-white/10 text-white/20 cursor-not-allowed'
                  : 'border-lens/40 text-lens-light hover:bg-lens/10 cursor-pointer'}`}>
              {downloading ? '⏳ Generating…' : '⬇ Download Report'}
            </button>
            <div className="text-xs font-mono text-white/30 glass rounded-xl px-4 py-2 border border-white/5">
              Audit ID: {carto?.audit_id}
            </div>
          </div>
        </div>
      </motion.div>

      {/* FairScore + Summary cards */}
      {carto?.fair_score && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }}
          className="mb-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
            <FairScoreGauge
              score={carto.fair_score.score}
              label={carto.fair_score.label}
              color={carto.fair_score.color}
            />
            <div className="md:col-span-3 grid grid-cols-2 md:grid-cols-2 gap-3 content-start">
              {[
                { label: 'Samples Analysed', value: carto?.summary?.total_samples?.toLocaleString(), color: 'text-white' },
                { label: 'Bias Hotspots', value: carto?.summary?.hotspot_count, color: 'text-signal-red' },
                { label: 'Decision Flips', value: constitution?.summary?.decision_flips, color: 'text-signal-amber' },
                { label: 'High-Risk Proxies', value: highRiskProxies || proxy?.summary?.critical_proxies, color: 'text-lens-light' },
              ].map((card, i) => (
                <div key={i} className="glass rounded-2xl p-4 border border-white/5">
                  <div className="text-white/30 text-xs font-mono mb-1">{card.label}</div>
                  <div className={`font-display font-bold text-2xl ${card.color}`}>{card.value ?? '—'}</div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* Summary cards (fallback when no fair_score) */}
      {!carto?.fair_score && <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
        {[
          { label: 'Samples Analysed', value: carto?.summary?.total_samples?.toLocaleString(), color: 'text-white' },
          { label: 'Bias Hotspots', value: carto?.summary?.hotspot_count, color: 'text-signal-red' },
          { label: 'Decision Flips', value: constitution?.summary?.decision_flips, color: 'text-signal-amber' },
          { label: 'High-Risk Proxies', value: highRiskProxies || proxy?.summary?.critical_proxies, color: 'text-lens-light' },
        ].map((card, i) => (
          <div key={i} className="glass rounded-2xl p-4 border border-white/5">
            <div className="text-white/30 text-xs font-mono mb-1">{card.label}</div>
            <div className={`font-display font-bold text-2xl ${card.color}`}>{card.value ?? '—'}</div>
          </div>
        ))}
      </motion.div>}

      {/* Compliance badges */}
      {carto?.compliance_tags?.length > 0 && (
        <ComplianceBadges tags={carto.compliance_tags} />
      )}

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
            <div className="space-y-4">
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
              {Object.keys(carto?.metric_confidence_intervals || {}).length > 0 && (
                <div className="glass rounded-2xl p-5 border border-white/5">
                  <h3 className="font-display font-semibold text-white text-sm mb-1">
                    Statistical Parity Difference — 95% Bootstrap Confidence Intervals
                  </h3>
                  <p className="text-white/30 text-xs mb-4 font-mono">
                    Bars show SPD per group. Whiskers show 95% CI from 200 bootstrap resamples. Bars above 0 = over-represented; below = under-represented.
                  </p>
                  <CIChart ciData={carto.metric_confidence_intervals} />
                </div>
              )}
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
