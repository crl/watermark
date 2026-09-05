import { ElectronAPI } from '@electron-toolkit/preload'
import type { WaterMarkAPI } from '../shared/types'

declare global {
  interface Window {
    electron: ElectronAPI
    api: WaterMarkAPI
  }
}

export {}
