import { create } from 'zustand'

export interface AuditSession {
  auditId: string | null
  modelFile: File | null
  datasetFile: File | null
  protectedCols: string[]
  targetCol: string
  cartographyResults: any | null
  constitutionResults: any | null
  proxyResults: any | null
  redteamResults: any | null
  confirmedBiases: any[]
  stage: 'upload' | 'cartography' | 'constitution' | 'proxy' | 'review' | 'redteam' | 'done'
  loading: boolean
  error: string | null
}

interface AuditStore extends AuditSession {
  setModelFile: (f: File | null) => void
  setDatasetFile: (f: File | null) => void
  setProtectedCols: (cols: string[]) => void
  setTargetCol: (col: string) => void
  setCartographyResults: (r: any) => void
  setConstitutionResults: (r: any) => void
  setProxyResults: (r: any) => void
  setRedteamResults: (r: any) => void
  setConfirmedBiases: (b: any[]) => void
  setStage: (s: AuditSession['stage']) => void
  setLoading: (v: boolean) => void
  setError: (e: string | null) => void
  reset: () => void
}

const initial: AuditSession = {
  auditId: null,
  modelFile: null,
  datasetFile: null,
  protectedCols: [],
  targetCol: '',
  cartographyResults: null,
  constitutionResults: null,
  proxyResults: null,
  redteamResults: null,
  confirmedBiases: [],
  stage: 'upload',
  loading: false,
  error: null,
}

export const useAuditStore = create<AuditStore>((set) => ({
  ...initial,
  setModelFile: (modelFile) => set({ modelFile }),
  setDatasetFile: (datasetFile) => set({ datasetFile }),
  setProtectedCols: (protectedCols) => set({ protectedCols }),
  setTargetCol: (targetCol) => set({ targetCol }),
  setCartographyResults: (cartographyResults) => set({ cartographyResults }),
  setConstitutionResults: (constitutionResults) => set({ constitutionResults }),
  setProxyResults: (proxyResults) => set({ proxyResults }),
  setRedteamResults: (redteamResults) => set({ redteamResults }),
  setConfirmedBiases: (confirmedBiases) => set({ confirmedBiases }),
  setStage: (stage) => set({ stage }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  reset: () => set(initial),
}))
