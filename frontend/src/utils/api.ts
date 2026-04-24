/// <reference types="vite/client" />
const BASE = import.meta.env.VITE_API_BASE_URL || ''

export async function runCartography(
  modelFile: File | null,
  protectedCols: string[],
  targetCol: string,
  extra?: Record<string, string>,
): Promise<any> {
  const fd = new FormData()
  if (modelFile) fd.append('model_file', modelFile)
  fd.append('protected_cols', protectedCols.join(','))
  fd.append('target_col', targetCol)
  if (extra) Object.entries(extra).forEach(([k, v]) => v && fd.append(k, v))
  const res = await fetch(`${BASE}/api/v1/cartography/analyze`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function runConstitution(
  modelFile: File | null,
  protectedCols: string[],
  targetCol: string,
  cartographyResults: any,
  modelType: string = 'sklearn',
  apiEndpoint: string = '',
  llmApiKey: string = '',
  hfToken: string = '',
  testSuite: string = 'auto',
): Promise<any> {
  const fd = new FormData()
  if (modelFile) fd.append('model_file', modelFile)
  fd.append('protected_cols', protectedCols.length > 0 ? protectedCols.join(',') : 'auto')
  fd.append('target_col', targetCol || 'auto')
  fd.append('cartography_results', JSON.stringify(cartographyResults))
  fd.append('model_type', modelType)
  if (apiEndpoint) fd.append('api_endpoint', apiEndpoint)
  if (llmApiKey) fd.append('llm_api_key', llmApiKey)
  if (hfToken) fd.append('hf_token', hfToken)
  fd.append('test_suite', testSuite)
  const res = await fetch(`${BASE}/api/v1/constitution/generate`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function runProxyHunter(
  protectedCols: string[],
  targetCol: string,
  modelType: string = 'sklearn',
  testSuite: string = 'auto',
): Promise<any> {
  const fd = new FormData()
  fd.append('protected_cols', protectedCols.length > 0 ? protectedCols.join(',') : 'auto')
  fd.append('target_col', targetCol || 'auto')
  fd.append('model_type', modelType)
  fd.append('test_suite', testSuite)
  const res = await fetch(`${BASE}/api/v1/proxy/hunt`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function exportPdfReport(result: any): Promise<void> {
  const res = await fetch(`${BASE}/api/v1/reports/pdf`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(result),
  })
  if (!res.ok) throw new Error(await res.text())
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `fairlens-report-${result?.audit_id ?? 'audit'}.pdf`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export function streamRedTeam(
  modelFile: File | null,
  protectedCols: string[],
  targetCol: string,
  confirmedBiases: any[],
  auditResults: any,
  onEvent: (e: any) => void,
  modelType: string = 'sklearn',
  apiEndpoint: string = '',
  llmApiKey: string = '',
  hfToken: string = '',
  testSuite: string = 'auto',
): () => void {
  const fd = new FormData()
  if (modelFile) fd.append('model_file', modelFile)
  fd.append('protected_cols', protectedCols.length > 0 ? protectedCols.join(',') : 'auto')
  fd.append('target_col', targetCol || 'auto')
  fd.append('confirmed_biases', JSON.stringify(confirmedBiases))
  const auditSummary = {
    cartography: {
      summary: auditResults?.cartography?.summary,
      audit_id: auditResults?.cartography?.audit_id,
      slice_metrics: auditResults?.cartography?.slice_metrics ?? [],
    },
    constitution: { summary: auditResults?.constitution?.summary },
    proxy: { summary: auditResults?.proxy?.summary },
  }
  fd.append('audit_results', JSON.stringify(auditSummary))
  fd.append('model_type', modelType)
  if (apiEndpoint) fd.append('api_endpoint', apiEndpoint)
  if (llmApiKey) fd.append('llm_api_key', llmApiKey)
  if (hfToken) fd.append('hf_token', hfToken)
  fd.append('test_suite', testSuite)

  const controller = new AbortController()
  fetch(`${BASE}/api/v1/redteam/run`, { method: 'POST', body: fd, signal: controller.signal })
    .then(async (res) => {
      if (!res.ok) {
        const errText = await res.text()
        onEvent({ node: 'error', status: 'error', log: [`Error: ${errText}`] })
        return
      }
      const reader = res.body!.getReader()
      const dec = new TextDecoder()
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        for (const line of dec.decode(value).split('\n')) {
          if (line.startsWith('data: ')) {
            try { onEvent(JSON.parse(line.slice(6))) } catch {}
          }
        }
      }
    }).catch(() => {})
  return () => controller.abort()
}
