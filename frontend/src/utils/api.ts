/// <reference types="vite/client" />
const BASE = import.meta.env.VITE_API_BASE_URL || ''

export async function runCartography(
  modelFile: File | null,
  datasetFile: File | null,
  protectedCols: string[],
  targetCol: string,
  extra?: Record<string, string>,
): Promise<any> {
  const fd = new FormData()
  if (modelFile) fd.append('model_file', modelFile)
  if (datasetFile) fd.append('dataset_file', datasetFile)
  fd.append('protected_cols', protectedCols.join(','))
  fd.append('target_col', targetCol)
  if (extra) Object.entries(extra).forEach(([k, v]) => v && fd.append(k, v))
  const res = await fetch(`${BASE}/api/v1/cartography/analyze`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function runConstitution(
  modelFile: File | null,
  datasetFile: File | null,
  protectedCols: string[],
  targetCol: string,
  cartographyResults: any,
  datasetSource: string = 'upload',
  datasetUrl: string = '',
  modelType: string = 'sklearn',
  apiEndpoint: string = '',
): Promise<any> {
  const fd = new FormData()
  if (modelFile) fd.append('model_file', modelFile)
  if (datasetFile) fd.append('dataset_file', datasetFile)
  fd.append('protected_cols', protectedCols.length > 0 ? protectedCols.join(',') : 'auto')
  fd.append('target_col', targetCol || 'auto')
  fd.append('cartography_results', JSON.stringify(cartographyResults))
  fd.append('dataset_source', datasetSource)
  if (datasetUrl) fd.append('dataset_url', datasetUrl)
  fd.append('model_type', modelType)
  if (apiEndpoint) fd.append('api_endpoint', apiEndpoint)
  const res = await fetch(`${BASE}/api/v1/constitution/generate`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function runProxyHunter(
  datasetFile: File | null,
  protectedCols: string[],
  targetCol: string,
  datasetSource: string = 'upload',
  datasetUrl: string = '',
): Promise<any> {
  const fd = new FormData()
  if (datasetFile) fd.append('dataset_file', datasetFile)
  fd.append('protected_cols', protectedCols.length > 0 ? protectedCols.join(',') : 'auto')
  fd.append('target_col', targetCol || 'auto')
  fd.append('dataset_source', datasetSource)
  if (datasetUrl) fd.append('dataset_url', datasetUrl)
  const res = await fetch(`${BASE}/api/v1/proxy/hunt`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export function streamRedTeam(
  modelFile: File | null,
  datasetFile: File | null,
  protectedCols: string[],
  targetCol: string,
  confirmedBiases: any[],
  auditResults: any,
  onEvent: (e: any) => void,
  datasetSource: string = 'upload',
  datasetUrl: string = '',
): () => void {
  const fd = new FormData()
  if (modelFile) fd.append('model_file', modelFile)
  if (datasetFile) fd.append('dataset_file', datasetFile)
  fd.append('protected_cols', protectedCols.length > 0 ? protectedCols.join(',') : 'auto')
  fd.append('target_col', targetCol || 'auto')
  fd.append('confirmed_biases', JSON.stringify(confirmedBiases))
  fd.append('audit_results', JSON.stringify(auditResults))
  fd.append('dataset_source', datasetSource)
  if (datasetUrl) fd.append('dataset_url', datasetUrl)

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
