import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import * as d3 from 'd3'
import ReactMarkdown from 'react-markdown'
import { useAuditStore } from '../hooks/useAuditStore'
import { exportPdfReport } from '../utils/api'

// ── Bias Map (D3 scatter — per demographic slice) ─────────────────────────
// Each dot = one demographic group. x = SPD, y = positive rate.
// Red = biased/flagged, green = fair. Size ∝ group size.
function BiasMap({ points, hotspots }: { points: any[]; hotspots: any[] }) {
  const ref = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!ref.current || !points.length) return
    const svg = d3.select(ref.current)
    svg.selectAll('*').remove()

    const W = ref.current.clientWidth || 520
    const H = 360
    const margin = { top: 24, right: 24, bottom: 52, left: 60 }
    const w = W - margin.left - margin.right
    const h = H - margin.top - margin.bottom

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    // Domains — SPD on x, positive rate on y
    const xVals = points.map((d: any) => d.x)
    const xMax  = Math.max(Math.abs(d3.min(xVals) as number), Math.abs(d3.max(xVals) as number), 0.05)
    const xScale = d3.scaleLinear().domain([-xMax * 1.2, xMax * 1.2]).range([0, w])
    const yScale = d3.scaleLinear().domain([0, 1]).range([h, 0])

    // Size scale: circle radius proportional to group size
    const maxSz = d3.max(points, (d: any) => d.size as number) as number || 1
    const rScale = d3.scaleSqrt().domain([0, maxSz]).range([3, 14])

    // Zero-line (x = 0 = no disparity)
    g.append('line')
      .attr('x1', xScale(0)).attr('x2', xScale(0))
      .attr('y1', 0).attr('y2', h)
      .attr('stroke', 'rgba(255,255,255,0.15)')
      .attr('stroke-dasharray', '5 3')

    // Overall positive rate reference line (first point's overall_rate if available)
    const overallRate = points[0]?.overall_rate
    if (overallRate != null) {
      g.append('line')
        .attr('x1', 0).attr('x2', w)
        .attr('y1', yScale(overallRate)).attr('y2', yScale(overallRate))
        .attr('stroke', 'rgba(255,255,255,0.1)')
        .attr('stroke-dasharray', '5 3')
      g.append('text')
        .attr('x', w + 4).attr('y', yScale(overallRate) + 3)
        .attr('fill', 'rgba(255,255,255,0.25)').attr('font-size', 7.5)
        .attr('font-family', 'JetBrains Mono, monospace')
        .text(`avg ${(overallRate * 100).toFixed(0)}%`)
    }

    // Axes
    g.append('g').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat((d: any) => `${(+d * 100).toFixed(0)}%`))
      .call(ax => ax.select('.domain').remove())
      .selectAll('text').attr('fill', '#ffffff55').attr('font-size', 8.5).attr('font-family', 'JetBrains Mono, monospace')

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5).tickFormat((d: any) => `${(+d * 100).toFixed(0)}%`))
      .call(ax => ax.select('.domain').remove())
      .selectAll('text').attr('fill', '#ffffff55').attr('font-size', 8.5).attr('font-family', 'JetBrains Mono, monospace')

    // Axis labels
    svg.append('text')
      .attr('x', margin.left + w / 2).attr('y', H - 8)
      .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.3)')
      .attr('font-size', 9).attr('font-family', 'JetBrains Mono, monospace')
      .text('Statistical Parity Difference  ←  disadvantaged  |  advantaged  →')

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(margin.top + h / 2)).attr('y', 14)
      .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.3)')
      .attr('font-size', 9).attr('font-family', 'JetBrains Mono, monospace')
      .text('Positive Rate')

    // Tooltip div
    const tooltip = d3.select('body').append('div')
      .style('position', 'fixed').style('pointer-events', 'none')
      .style('background', '#1e293b').style('border', '1px solid rgba(255,255,255,0.1)')
      .style('border-radius', '8px').style('padding', '6px 10px')
      .style('font-size', '11px').style('font-family', 'JetBrains Mono, monospace')
      .style('color', '#e2e8f0').style('z-index', '9999').style('opacity', '0')
      .style('max-width', '220px').style('line-height', '1.5')

    // Points
    g.selectAll('circle.pt')
      .data(points)
      .join('circle')
      .attr('class', 'pt')
      .attr('cx', (d: any) => xScale(d.x))
      .attr('cy', (d: any) => yScale(d.y))
      .attr('r', (d: any) => rScale(d.size || 1))
      .attr('fill', (d: any) => d.flagged ? '#ef4444' : '#22c55e')
      .attr('fill-opacity', (d: any) => d.flagged ? 0.75 : 0.55)
      .attr('stroke', (d: any) => d.flagged ? '#ef4444' : '#22c55e')
      .attr('stroke-opacity', 0.9)
      .attr('stroke-width', 1)
      .on('mouseover', function(event: MouseEvent, d: any) {
        d3.select(this).attr('fill-opacity', 1).attr('r', rScale(d.size || 1) + 2)
        tooltip.style('opacity', '1')
          .html(`<b>${d.slice_label}</b><br/>SPD: ${(d.x * 100).toFixed(1)}%<br/>Pos. rate: ${(d.y * 100).toFixed(1)}%<br/>n = ${d.size}`)
      })
      .on('mousemove', function(event: MouseEvent) {
        tooltip.style('left', `${event.clientX + 12}px`).style('top', `${event.clientY - 28}px`)
      })
      .on('mouseout', function(_event: any, d: any) {
        d3.select(this).attr('fill-opacity', d.flagged ? 0.75 : 0.55).attr('r', rScale(d.size || 1))
        tooltip.style('opacity', '0')
      })

    // Labels for flagged points
    points.filter((d: any) => d.flagged).forEach((d: any) => {
      const shortLabel = d.slice_label.length > 18 ? d.slice_label.slice(0, 18) + '…' : d.slice_label
      g.append('text')
        .attr('x', xScale(d.x))
        .attr('y', yScale(d.y) - rScale(d.size || 1) - 3)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ef4444cc')
        .attr('font-size', 8)
        .attr('font-family', 'JetBrains Mono, monospace')
        .text(shortLabel)
    })

    return () => { tooltip.remove() }
  }, [points, hotspots])

  return (
    <svg ref={ref} width="100%" height={360}
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
  const [activeTab, setActiveTab] = useState<'cartography' | 'constitution' | 'proxy' | 'model_probe' | 'dataset_probe' | 'cross_synthesis'>('model_probe')
  const [selectedBiases, setSelectedBiases] = useState<Set<string>>(new Set())
  const [downloading, setDownloading] = useState(false)

  const downloadReport = async () => {
    if (!store.crossAnalysisResults && !store.cartographyResults) return
    setDownloading(true)
    try {
      const fullReport = {
        ...(store.crossAnalysisResults?.cartography ?? store.cartographyResults),
        constitution:     store.crossAnalysisResults?.constitution ?? store.constitutionResults ?? null,
        proxy_hunt:       store.crossAnalysisResults?.proxy ?? store.proxyResults ?? null,
        model_probe:      store.modelProbeResults ?? null,
        dataset_probe:    store.datasetProbeResults ?? null,
        cross_synthesis:  store.crossAnalysisResults?.cross_synthesis ?? null,
        redteam:          store.redteamResults ?? null,
      }
      await exportPdfReport(fullReport)
    } catch (e: any) {
      alert(`PDF export failed: ${e.message}`)
    } finally {
      setDownloading(false)
    }
  }

  // Phase results
  const modelProbe   = store.modelProbeResults
  const datasetProbe = store.datasetProbeResults
  const crossResult  = store.crossAnalysisResults

  // Cross-analysis stages (Phase 3)
  const carto        = crossResult?.cartography ?? store.cartographyResults
  const constitution = crossResult?.constitution ?? store.constitutionResults
  const proxy        = crossResult?.proxy ?? store.proxyResults
  const crossSynth   = crossResult?.cross_synthesis

  if (!carto && !modelProbe && !datasetProbe) {
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

  // Biases from Phase 1 (model probe)
  const modelProbeBiases: any[] = modelProbe?.model_biases ?? []
  const modelProbeDiagnostics = modelProbe?.summary?.prediction_diagnostics ?? modelProbe?.prediction_diagnostics ?? null
  // Biases from Phase 2 (dataset probe)
  const datasetProbeBiases: any[] = datasetProbe?.dataset_biases ?? []

  // Full confirmed bias list across all phases
  const allBiases = [
    // Phase 1 — model probe biases
    ...modelProbeBiases.map((b: any) => ({
      attribute: b.attribute,
      type:      b.type,
      label:     `[Model Probe] ${b.value || b.attribute}`,
      magnitude: b.magnitude,
      source:    'model_probe',
      _raw:      b,
    })),
    // Phase 2 — dataset probe biases
    ...datasetProbeBiases.map((b: any) => ({
      attribute: b.attribute,
      type:      b.type,
      label:     `[Dataset Probe] ${b.value || b.attribute}`,
      magnitude: b.magnitude,
      source:    'dataset_probe',
      _raw:      b,
    })),
    // Phase 3 — cross-analysis flagged slices
    ...(carto?.slice_metrics?.filter((m: any) => m.flagged).map((m: any) => ({
      attribute: m.attribute || 'unknown',
      type:      'demographic_parity',
      label:     `[Cross-Analysis] ${m.label}`,
      spd:       m.statistical_parity_diff,
      magnitude: Math.abs(m.statistical_parity_diff),
      source:    'cross_analysis',
      _raw:      m,
    })) ?? []),
    // Phase 3 — high-risk proxy chains
    ...(proxy?.proxy_chains?.filter((c: any) => c.risk_level === 'critical' || c.risk_level === 'high').map((c: any) => ({
      attribute: c.start_feature,
      type:      'proxy',
      label:     `[Proxy] ${c.explanation ?? c.path?.join(' → ')}`,
      chain:     c.path,
      magnitude: c.risk_score ?? 0,
      source:    'cross_analysis_proxy',
      _raw:      c,
    })) ?? []),
  ]

  const selectedList = allBiases.filter((b: any) => selectedBiases.has(b.label))

  const proceedToRedTeam = () => {
    store.setConfirmedBiases(selectedList)
    store.setStage('redteam')
    nav('/redteam')
  }

  const highRiskProxies = (proxy?.summary?.critical_proxies ?? 0) + (proxy?.summary?.high_proxies ?? 0)

  // Merge compliance tags from all three phases — show worst-case status per regulation
  const mergedComplianceTags = (() => {
    const allTags = [
      ...(modelProbe?.cartography?.compliance_tags ?? []),
      ...(datasetProbe?.cartography?.compliance_tags ?? []),
      ...(carto?.compliance_tags ?? []),
    ]
    if (!allTags.length) return []
    // Group by regulation ID and pick worst status (FAIL > CAUTION > PASS)
    const statusRank: Record<string, number> = { FAIL: 2, CAUTION: 1, PASS: 0 }
    const byId: Record<string, any> = {}
    for (const tag of allTags) {
      const existing = byId[tag.id]
      if (!existing || statusRank[tag.status] > statusRank[existing.status]) {
        // Merge violations from all phases
        byId[tag.id] = {
          ...tag,
          violations: [...(existing?.violations ?? []), ...(tag.violations ?? [])].filter((v, i, arr) => arr.indexOf(v) === i),
          worst_spd: Math.max(existing?.worst_spd ?? 0, tag.worst_spd ?? 0),
          worst_di: Math.min(existing?.worst_di ?? 1, tag.worst_di ?? 1),
        }
      }
    }
    return Object.values(byId)
  })()

  // Adjust FairScore to account for proxy chain risk (cartography score only sees model outputs)
  const adjustedFairScore = (() => {
    if (!carto?.fair_score) return null
    let score = carto.fair_score.score
    score -= (proxy?.summary?.critical_proxies ?? 0) * 10
    score -= (proxy?.summary?.high_proxies ?? 0) * 5
    score = Math.max(0, Math.min(100, score))
    const label = score >= 80 ? 'Fair' : score >= 60 ? 'Caution' : 'Biased'
    const color = score >= 80 ? 'green' : score >= 60 ? 'yellow' : 'red'
    return { score, label, color }
  })()

  const TABS = [
    { key: 'model_probe',  label: 'Phase 1 — Model',   count: modelProbeBiases.length },
    { key: 'dataset_probe', label: 'Phase 2 — Dataset', count: datasetProbeBiases.length },
    { key: 'cross_synthesis', label: 'Phase 3 — Cross', count: crossSynth?.summary?.total_compounded_risks ?? 0 },
    { key: 'cartography', label: 'Bias Map',    count: carto?.slice_metrics?.filter((m: any) => m.flagged)?.length ?? 0 },
    { key: 'constitution', label: 'Constitution', count: constitution?.patterns?.length ?? 0 },
    { key: 'proxy',        label: 'Proxy Chains', count: highRiskProxies },
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
      {adjustedFairScore && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }}
          className="mb-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
            <FairScoreGauge
              score={adjustedFairScore.score}
              label={adjustedFairScore.label}
              color={adjustedFairScore.color}
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

      {/* Compliance badges — worst-case across all phases */}
      {mergedComplianceTags.length > 0 && (
        <ComplianceBadges tags={mergedComplianceTags} />
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

          {/* ── Phase 1: Model Probe ───────────────────────────────────────── */}
          {activeTab === 'model_probe' && (
            <div className="space-y-4">
              {modelProbe ? (
                <>
                  {(modelProbeDiagnostics?.collapsed_output || modelProbeDiagnostics?.near_constant_output) && (
                    <div className="glass rounded-2xl p-5 border border-signal-red/20 bg-signal-red/5">
                      <div className="text-xs font-mono text-signal-red mb-2">Probe Validity Warning</div>
                      <p className="text-white/80 text-sm">{modelProbeDiagnostics?.reason}</p>
                      <p className="text-white/40 text-xs mt-2">
                        Phase 1 fairness metrics are not trustworthy when the model outputs an almost constant prediction on the reference probe.
                      </p>
                    </div>
                  )}
                  <div className="grid md:grid-cols-3 gap-4">
                    <div className="glass rounded-2xl p-5 border border-lens/15">
                      <div className="text-xs font-mono text-white/40 mb-2">Model FairScore™ (Reference Dataset)</div>
                      <div className={`font-display font-bold text-4xl ${
                        (modelProbe.summary?.fair_score?.color === 'green') ? 'text-green-400' :
                        (modelProbe.summary?.fair_score?.color === 'yellow') ? 'text-yellow-400' : 'text-red-400'
                      }`}>{modelProbe.summary?.fair_score?.score ?? '—'}</div>
                      <div className="text-white/30 text-xs mt-1">{modelProbe.summary?.fair_score?.label} · {modelProbe.reference_dataset_size} reference rows</div>
                    </div>
                    <div className="glass rounded-2xl p-5 border border-white/5">
                      <div className="text-xs font-mono text-white/40 mb-2">Hidden Biases Detected</div>
                      <div className="text-signal-red font-display font-bold text-4xl">{modelProbeBiases.length}</div>
                      <div className="text-white/30 text-xs mt-1">intrinsic model biases</div>
                    </div>
                    <div className="glass rounded-2xl p-5 border border-white/5">
                      <div className="text-xs font-mono text-white/40 mb-2">Most Biased Attribute</div>
                      <div className="text-lens-light font-display font-bold text-xl">{modelProbe.summary?.most_biased_attribute ?? '—'}</div>
                      <div className="text-white/30 text-xs mt-1">on reference probe</div>
                    </div>
                  </div>
                  <div className="glass rounded-2xl p-5 border border-white/5">
                    <div className="text-xs font-mono text-white/40 uppercase tracking-widest mb-3">Detected Model Biases</div>
                    <div className="space-y-2">
                      {modelProbeBiases.map((b: any, i: number) => (
                        <div key={i} className="flex items-center justify-between p-3 rounded-xl bg-white/2 border border-white/5">
                          <div>
                            <span className="text-white/80 text-xs font-mono font-semibold">{b.attribute}</span>
                            <span className="text-white/40 text-xs ml-2">{b.value}</span>
                          </div>
                          <p className="text-white/60 text-sm leading-relaxed mb-3">{modelProbe.degenerate_message}</p>
                          <div className="text-white/40 text-xs font-mono space-y-1">
                            <div>› Reference probe size: {modelProbe.reference_dataset_size} rows</div>
                            <div>› Protected cols tested: {modelProbe.reference_protected_cols?.join(', ')}</div>
                            <div>› What this means: the model was not trained on data with demographic features matching our reference dataset, so it collapses all predictions to one class on synthetic inputs.</div>
                            <div>› What to do: upload a representative dataset — Phase 3 Cross-Analysis will then measure real demographic bias using your actual data.</div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="glass rounded-2xl p-5 border border-white/5 text-center">
                      <div className="text-white/20 text-xs font-mono">FairScore™ not available — probe inconclusive</div>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="grid md:grid-cols-3 gap-4">
                      <div className="glass rounded-2xl p-5 border border-lens/15">
                        <div className="text-xs font-mono text-white/40 mb-2">Model FairScore™ (Reference Dataset)</div>
                        <div className={`font-display font-bold text-4xl ${
                          (modelProbe.summary?.fair_score?.color === 'green') ? 'text-green-400' :
                          (modelProbe.summary?.fair_score?.color === 'yellow') ? 'text-yellow-400' : 'text-red-400'
                        }`}>{modelProbe.summary?.fair_score?.score ?? '—'}</div>
                        <div className="text-white/30 text-xs mt-1">{modelProbe.summary?.fair_score?.label} · {modelProbe.reference_dataset_size} reference rows</div>
                      </div>
                      <div className="glass rounded-2xl p-5 border border-white/5">
                        <div className="text-xs font-mono text-white/40 mb-2">Hidden Biases Detected</div>
                        <div className="text-signal-red font-display font-bold text-4xl">{modelProbeBiases.length}</div>
                        <div className="text-white/30 text-xs mt-1">intrinsic model biases</div>
                      </div>
                      <div className="glass rounded-2xl p-5 border border-white/5">
                        <div className="text-xs font-mono text-white/40 mb-2">Most Biased Attribute</div>
                        <div className="text-lens-light font-display font-bold text-xl">{modelProbe.summary?.most_biased_attribute ?? '—'}</div>
                        <div className="text-white/30 text-xs mt-1">on reference probe</div>
                      </div>
                    </div>
                    <div className="glass rounded-2xl p-5 border border-white/5">
                      <div className="text-xs font-mono text-white/40 uppercase tracking-widest mb-3">Detected Model Biases</div>
                      <div className="space-y-2">
                        {modelProbeBiases.map((b: any, i: number) => (
                          <div key={i} className="flex items-center justify-between p-3 rounded-xl bg-white/2 border border-white/5">
                            <div>
                              <span className="text-white/80 text-xs font-mono font-semibold">{b.attribute}</span>
                              <span className="text-white/40 text-xs ml-2">{b.value}</span>
                            </div>
                            <div className="flex items-center gap-3">
                              <span className="text-white/30 text-xs font-mono">magnitude {b.magnitude?.toFixed(3)}</span>
                              <SeverityBadge level={b.severity} />
                            </div>
                          </div>
                        ))}
                        {modelProbeBiases.length === 0 && <div className="text-white/20 text-sm font-mono text-center py-4">No hidden biases detected in model probe</div>}
                      </div>
                    </div>
                    {modelProbe.cartography?.gemini_analysis?.headline && (
                      <div className="glass rounded-2xl p-5 border border-lens/15">
                        <div className="text-xs font-mono text-lens-light mb-2">Gemini Analysis — Model Probe</div>
                        <p className="text-white/80 text-sm mb-2">{modelProbe.cartography.gemini_analysis.headline}</p>
                        <ul className="space-y-1">
                          {modelProbe.cartography.gemini_analysis.key_findings?.map((f: string, i: number) => (
                            <li key={i} className="text-white/50 text-xs font-mono flex gap-2"><span className="text-lens-light">›</span>{f}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </>
                )
              ) : (
                <div className="glass rounded-2xl p-8 text-center text-white/30 text-sm font-mono border border-white/5">
                  Model probe results not available
                </div>
              )}
            </div>
          )}

          {/* ── Phase 2: Dataset Probe ─────────────────────────────────────── */}
          {activeTab === 'dataset_probe' && (
            <div className="space-y-4">
              {datasetProbe ? (
                <>
                  <div className="grid md:grid-cols-3 gap-4">
                    <div className="glass rounded-2xl p-5 border border-lens/15">
                      <div className="text-xs font-mono text-white/40 mb-2">Dataset FairScore™ (No Model)</div>
                      <div className={`font-display font-bold text-4xl ${
                        (datasetProbe.summary?.fair_score?.color === 'green') ? 'text-green-400' :
                        (datasetProbe.summary?.fair_score?.color === 'yellow') ? 'text-yellow-400' : 'text-red-400'
                      }`}>{datasetProbe.summary?.fair_score?.score ?? '—'}</div>
                      <div className="text-white/30 text-xs mt-1">{datasetProbe.summary?.fair_score?.label}</div>
                    </div>
                    <div className="glass rounded-2xl p-5 border border-white/5">
                      <div className="text-xs font-mono text-white/40 mb-2">Dataset Biases</div>
                      <div className="text-signal-amber font-display font-bold text-4xl">{datasetProbeBiases.length}</div>
                      <div className="text-white/30 text-xs mt-1">structural biases in data</div>
                    </div>
                    <div className="glass rounded-2xl p-5 border border-white/5">
                      <div className="text-xs font-mono text-white/40 mb-2">Critical Proxy Chains</div>
                      <div className="text-signal-red font-display font-bold text-4xl">{datasetProbe.summary?.critical_proxies ?? 0}</div>
                      <div className="text-white/30 text-xs mt-1">out of {datasetProbe.summary?.proxy_count ?? 0} total</div>
                    </div>
                  </div>
                  <div className="glass rounded-2xl p-5 border border-white/5">
                    <div className="text-xs font-mono text-white/40 uppercase tracking-widest mb-3">Detected Dataset Biases</div>
                    <div className="space-y-2">
                      {datasetProbeBiases.map((b: any, i: number) => (
                        <div key={i} className="flex items-center justify-between p-3 rounded-xl bg-white/2 border border-white/5">
                          <div>
                            <span className="text-white/80 text-xs font-mono font-semibold">{b.attribute}</span>
                            <span className="text-white/40 text-xs ml-2">{b.type === 'proxy_chain' ? `→ ${b.value}` : b.value}</span>
                          </div>
                          <div className="flex items-center gap-3">
                            <span className="text-white/30 text-xs font-mono">magnitude {b.magnitude?.toFixed(3)}</span>
                            <SeverityBadge level={b.severity} />
                          </div>
                        </div>
                      ))}
                      {datasetProbeBiases.length === 0 && <div className="text-white/20 text-sm font-mono text-center py-4">No structural biases detected in dataset probe</div>}
                    </div>
                  </div>
                </>
              ) : (
                <div className="glass rounded-2xl p-8 text-center text-white/30 text-sm font-mono border border-white/5">
                  Dataset probe results not available
                </div>
              )}
            </div>
          )}

          {/* ── Phase 3: Cross-Analysis Synthesis ─────────────────────────── */}
          {activeTab === 'cross_synthesis' && (
            <div className="space-y-4">
              {crossSynth ? (
                <>
                  {crossSynth.gemini_analysis?.headline && (
                    <div className="glass rounded-2xl p-5 border border-lens/20">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-lens-light text-xs font-mono font-semibold">Cross-Analysis AI Synthesis</span>
                        <SeverityBadge level={crossSynth.gemini_analysis.severity} />
                      </div>
                      <p className="text-white/80 text-sm mb-2">{crossSynth.gemini_analysis.headline}</p>
                      {crossSynth.gemini_analysis.interaction_mechanism && (
                        <p className="text-white/50 text-xs mb-3">{crossSynth.gemini_analysis.interaction_mechanism}</p>
                      )}
                      <ul className="space-y-1">
                        {crossSynth.gemini_analysis.key_findings?.map((f: string, i: number) => (
                          <li key={i} className="text-white/50 text-xs font-mono flex gap-2"><span className="text-lens-light">›</span>{f}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  <div className="grid md:grid-cols-3 gap-4">
                    {[
                      { label: 'Aligned Biases', value: crossSynth.summary?.aligned_count, desc: 'in both model & dataset', color: 'text-signal-red' },
                      { label: 'Proxy Amplifications', value: crossSynth.summary?.proxy_amplification_count, desc: 'proxy → biased attribute', color: 'text-signal-amber' },
                      { label: 'Blind Spots', value: crossSynth.summary?.blind_spot_count, desc: 'one-sided bias', color: 'text-lens-light' },
                    ].map((c, i) => (
                      <div key={i} className="glass rounded-2xl p-5 border border-white/5">
                        <div className="text-xs font-mono text-white/40 mb-1">{c.label}</div>
                        <div className={`font-display font-bold text-3xl ${c.color}`}>{c.value ?? '—'}</div>
                        <div className="text-white/25 text-xs mt-1">{c.desc}</div>
                      </div>
                    ))}
                  </div>
                  {crossSynth.risk_matrix?.length > 0 && (
                    <div className="glass rounded-2xl p-5 border border-white/5">
                      <div className="text-xs font-mono text-white/40 uppercase tracking-widest mb-3">Risk Matrix</div>
                      <div className="space-y-2">
                        {crossSynth.risk_matrix.map((r: any, i: number) => (
                          <div key={i} className="p-3 rounded-xl bg-white/2 border border-white/5">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-white/80 text-xs font-mono font-semibold">{r.attribute}</span>
                              <div className="flex items-center gap-2">
                                <span className="text-white/30 text-xs font-mono">{r.risk_type}</span>
                                <SeverityBadge level={r.severity} />
                              </div>
                            </div>
                            <p className="text-white/40 text-xs">{r.description}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="glass rounded-2xl p-8 text-center text-white/30 text-sm font-mono border border-white/5">
                  Cross-analysis synthesis not available
                </div>
              )}
            </div>
          )}

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
            <div className="space-y-6">
              {/* Gemini narrative */}
              {proxy.gemini_analysis?.headline && (
                <div className="glass rounded-2xl p-5 border border-lens/20">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-lens-light text-xs font-mono font-semibold">AI Analysis</span>
                    <SeverityBadge level={proxy.gemini_analysis.severity} />
                  </div>
                  <p className="text-white/80 text-sm mb-3">{proxy.gemini_analysis.headline}</p>
                  {proxy.gemini_analysis.key_findings?.length > 0 && (
                    <ul className="space-y-1 mb-3">
                      {proxy.gemini_analysis.key_findings.map((f: string, i: number) => (
                        <li key={i} className="text-white/50 text-xs font-mono flex gap-2">
                          <span className="text-lens-light">›</span>{f}
                        </li>
                      ))}
                    </ul>
                  )}
                  {proxy.gemini_analysis.debiasing_strategy && (
                    <div className="mt-2 p-3 rounded-xl bg-white/3 border border-white/5">
                      <span className="text-white/40 text-xs font-mono">Debiasing strategy: </span>
                      <span className="text-white/70 text-xs">{proxy.gemini_analysis.debiasing_strategy}</span>
                    </div>
                  )}
                </div>
              )}
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
                      <div className="text-white/30 text-xs mb-1">{c.explanation}</div>
                      {(c.corr_with_protected != null || c.corr_with_target != null) && (
                        <div className="flex gap-3 mt-1">
                          {c.corr_with_protected != null && (
                            <span className="text-white/30 text-xs font-mono">
                              protected corr: <span className="text-signal-amber">{(c.corr_with_protected * 100).toFixed(0)}%</span>
                            </span>
                          )}
                          {c.corr_with_target != null && (
                            <span className="text-white/30 text-xs font-mono">
                              target corr: <span className="text-signal-red">{(c.corr_with_target * 100).toFixed(0)}%</span>
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
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
          {allBiases.map((b: any, i: number) => {
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
          {allBiases.length === 0 && (
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
