import { app, shell, BrowserWindow, ipcMain, dialog, protocol } from 'electron'
import { join, extname } from 'path'
import { createReadStream } from 'fs'
import { copyFile, stat } from 'fs/promises'
import { Readable } from 'stream'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { Sidecar } from './sidecar'

const sidecar = new Sidecar()

function mimeForVideo(filePath: string): string {
  switch (extname(filePath).toLowerCase()) {
    case '.mp4':
    case '.m4v':
      return 'video/mp4'
    case '.webm':
      return 'video/webm'
    case '.mov':
      return 'video/quicktime'
    case '.mkv':
      return 'video/x-matroska'
    case '.avi':
      return 'video/x-msvideo'
    default:
      return 'application/octet-stream'
  }
}

function parseByteRange(header: string | null, size: number): { start: number; end: number } | null {
  if (!header) return null
  const match = /bytes=(\d*)-(\d*)/.exec(header)
  if (!match) return null
  const start = match[1] === '' ? 0 : Number(match[1])
  const end = match[2] === '' ? size - 1 : Number(match[2])
  if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || start >= size || end < start) {
    return null
  }
  return { start, end: Math.min(end, size - 1) }
}

async function handleMediaRequest(request: Request): Promise<Response> {
  const filePath = new URL(request.url).searchParams.get('path')
  if (!filePath) {
    return new Response('missing path', { status: 400 })
  }
  let fileSize = 0
  try {
    fileSize = (await stat(filePath)).size
  } catch {
    return new Response('not found', { status: 404 })
  }

  const type = mimeForVideo(filePath)
  const range = parseByteRange(request.headers.get('range'), fileSize)
  const headers = new Headers({
    'Content-Type': type,
    'Accept-Ranges': 'bytes',
    'Cache-Control': 'no-store'
  })

  if (request.method === 'HEAD') {
    headers.set('Content-Length', String(fileSize))
    return new Response(null, { status: 200, headers })
  }

  if (!range) {
    headers.set('Content-Length', String(fileSize))
    const stream = Readable.toWeb(createReadStream(filePath)) as ReadableStream<Uint8Array>
    return new Response(stream, { status: 200, headers })
  }

  const { start, end } = range
  headers.set('Content-Length', String(end - start + 1))
  headers.set('Content-Range', `bytes ${start}-${end}/${fileSize}`)
  const stream = Readable.toWeb(createReadStream(filePath, { start, end })) as ReadableStream<Uint8Array>
  return new Response(stream, { status: 206, headers })
}

protocol.registerSchemesAsPrivileged([
  {
    scheme: 'wmmedia',
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      stream: true,
      bypassCSP: true,
      corsEnabled: true
    }
  }
])

function createWindow(): void {
  const mainWindow = new BrowserWindow({
    width: 1360,
    height: 900,
    minWidth: 1100,
    minHeight: 740,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#09090b',
    title: 'WaterMark',
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

function broadcastSidecar(): void {
  for (const win of BrowserWindow.getAllWindows()) {
    win.webContents.send('sidecar:state', sidecar.state)
  }
}

app.whenReady().then(() => {
  protocol.handle('wmmedia', (request) => handleMediaRequest(request))

  electronApp.setAppUserModelId('com.watermark.app')

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  sidecar.onChange(() => broadcastSidecar())

  ipcMain.handle('sidecar:get-state', () => sidecar.state)
  ipcMain.handle('sidecar:get-api-base', () => sidecar.apiBase)
  ipcMain.handle('dialog:open-video', async () => {
    const result = await dialog.showOpenDialog({
      title: '选择视频',
      properties: ['openFile'],
      filters: [
        { name: '视频', extensions: ['mp4', 'mov', 'mkv', 'avi', 'webm', 'm4v'] },
        { name: '全部', extensions: ['*'] }
      ]
    })
    return result.canceled ? null : result.filePaths[0]
  })
  ipcMain.handle('dialog:save-video', async (_event, defaultName: string) => {
    const result = await dialog.showSaveDialog({
      title: '导出视频',
      defaultPath: defaultName,
      filters: [{ name: 'MP4', extensions: ['mp4'] }]
    })
    return result.canceled ? null : result.filePath
  })
  ipcMain.handle('shell:reveal', async (_event, target: string) => {
    shell.showItemInFolder(target)
  })
  ipcMain.handle('fs:copy', async (_event, from: string, to: string) => {
    await copyFile(from, to)
    return to
  })
  ipcMain.handle('media:url', (_event, target: string) => {
    return `wmmedia://local/?path=${encodeURIComponent(target)}`
  })

  createWindow()
  sidecar.start().catch(() => undefined)

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('before-quit', () => {
  sidecar.stop()
})

app.on('window-all-closed', () => {
  sidecar.stop()
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
