import { contextBridge, ipcRenderer, webUtils } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'
import type { SidecarState, WaterMarkAPI } from '../shared/types'

const api: WaterMarkAPI = {
  getSidecarState: () => ipcRenderer.invoke('sidecar:get-state'),
  getApiBase: () => ipcRenderer.invoke('sidecar:get-api-base'),
  onSidecarState: (callback) => {
    const listener = (_event: unknown, state: SidecarState): void => callback(state)
    ipcRenderer.on('sidecar:state', listener)
    return () => {
      ipcRenderer.removeListener('sidecar:state', listener)
    }
  },
  openVideo: () => ipcRenderer.invoke('dialog:open-video'),
  saveVideo: (defaultName: string) => ipcRenderer.invoke('dialog:save-video', defaultName),
  revealInFolder: (target: string) => ipcRenderer.invoke('shell:reveal', target),
  copyFile: (from: string, to: string) => ipcRenderer.invoke('fs:copy', from, to),
  toMediaUrl: (target: string) => ipcRenderer.invoke('media:url', target),
  getPathForFile: (file: File) => webUtils.getPathForFile(file)
}

if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('api', api)
  } catch (error) {
    console.error(error)
  }
} else {
  // @ts-ignore defined in dts
  window.electron = electronAPI
  // @ts-ignore defined in dts
  window.api = api
}
