/// <reference types="vite/client" />
const BASE = import.meta.env.VITE_API_BASE_URL || ''

export type ModelType = 'sklearn' | 'api' | 'huggingface' | 'vertex_ai'

export interface ModelConfig {
  modelType: ModelType
  modelFile?: File | null         // for sklearn
  apiEndpoint?: string            // for api / huggingface
  vertexEndpointId?: string       // for vertex_ai
  gcpProject?: string             // for vertex_ai
}

function buildModelFormData(fd: FormData, cfg: ModelConfig) {
  fd.append('model_type', cfg.modelType)
  if (cfg.modelFile) fd.append('model_file', cfg.modelFile)
  if (cfg.apiEndpoint) fd.append('api_endpoint', cfg.apiEndpoint)
  if (cfg.vertexEndpointId) fd.append('vertex_endpoint_id', cfg.vertexEndpointId)
  if (cfg.gcpProject) fd.append('gcp_project', cfg.gcpProject)
}

export async function runCartography(
  modelCfg: ModelConfig,
  datasetFile: File,
  protectedCols: string[],
  targetCol: string,
): Promise<any> {
  const fd = new FormData()
  buildModelFormData(fd, modelCfg)
  fd.append('dataset_file', datasetFile)
  fd.append('protected_cols', protectedCols.join(','))
  fd.append('target_col', targetCol)
  const res = await fetch(`${BASE}/api/v1/cartography/analyze`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error(`Cartography failed: ${await res.text()}`)
  return res.json()
}

export async function runConstitution(
  modelCfg: ModelConfig,
  datasetFile: File,
  protectedCols: string[],
  targetCol: string,
  cartographyResults: any,
): Promise<any> {
  const fd = new FormData()
  buildModelFormData(fd, modelCfg)
  fd.append('dataset_file', datasetFile)
  fd.append('protected_cols', protectedCols.join(','))
  fd.append('target_col', targetCol)
  fd.append('cartography_results', JSON.stringify(cartographyResults))
  const res = await fetch(`${BASE}/api/v1/constitution/generate`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error(`Constitution failed: ${await res.text()}`)
  return res.json()
}

export async function runProxyHunter(
  datasetFile: File,
  protectedCols: string[],
  targetCol: string,
): Promise<any> {
  const fd = new FormData()
  fd.append('dataset_file', datasetFile)
  fd.append('protected_cols', protectedCols.join(','))
  fd.append('target_col', targetCol)
  const res = await fetch(`${BASE}/api/v1/proxy/hunt`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error(`Proxy hunt failed: ${await res.text()}`)
  return res.json()
}

export function streamRedTeam(
  modelCfg: ModelConfig,
  datasetFile: File,
  protectedCols: string[],
  targetCol: string,
  confirmedBiases: any[],
  auditResults: any,
  onEvent: (e: any) => void,
): () => void {
  const fd = new FormData()
  buildModelFormData(fd, modelCfg)
  fd.append('dataset_file', datasetFile)
  fd.append('protected_cols', protectedCols.join(','))
  fd.append('target_col', targetCol)
  fd.append('confirmed_biases', JSON.stringify(confirmedBiases))
  fd.append('audit_results', JSON.stringify(auditResults))

  const controller = new AbortController()
  fetch(`${BASE}/api/v1/redteam/run`, { method: 'POST', body: fd, signal: controller.signal })
    .then(async (res) => {
      const reader = res.body!.getReader()
      const dec = new TextDecoder()
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const text = dec.decode(value)
        for (const line of text.split('\n')) {
          if (line.startsWith('data: ')) {
            try { onEvent(JSON.parse(line.slice(6))) } catch {}
          }
        }
      }
    })
    .catch(() => {})
  return () => controller.abort()
}
