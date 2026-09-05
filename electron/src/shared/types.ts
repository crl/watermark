export type SidecarState = {
  status: 'starting' | 'ready' | 'error' | 'stopped'
  port: number | null
  error: string | null
  python: string | null
}

export type HealthInfo = {
  ok: boolean
  ffmpeg: string | null
  ffmpegReady: boolean
  propainterReady: boolean
  cuda: {
    torch: boolean
    cuda: boolean
    device: string | null
    version: string | null
  }
  python: string
}

export type Rect = {
  x: number
  y: number
  width: number
  height: number
  rotation?: number
}

export type WaterMarkAPI = {
  getSidecarState: () => Promise<SidecarState>
  getApiBase: () => Promise<string | null>
  onSidecarState: (callback: (state: SidecarState) => void) => () => void
  openVideo: () => Promise<string | null>
  saveVideo: (defaultName: string) => Promise<string | null>
  revealInFolder: (target: string) => Promise<void>
  copyFile: (from: string, to: string) => Promise<string>
  toMediaUrl: (target: string) => Promise<string>
  getPathForFile: (file: File) => string
}
