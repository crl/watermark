import { useCallback, useEffect, useMemo, useRef, useState, type JSX } from 'react'
import { CompareSlider } from './components/CompareSlider'
import { TIME_EPS, VideoCanvas, type Region } from './components/VideoCanvas'
import type { HealthInfo, Rect, SidecarState } from '@shared/types'

type JobEvent = {
  stage: string
  progress?: number | null
  message?: string
  outputPath?: string
  engine?: string
  partial?: boolean
}

const VIDEO_EXT = ['.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v']

function basename(filePath: string): string {
  return filePath.replace(/\\/g, '/').split('/').pop() || filePath
}

function stem(filePath: string): string {
  return basename(filePath).replace(/\.[^.]+$/, '')
}

function groupByTime(regions: Region[]): Array<{ timeSec: number; items: Region[] }> {
  const sorted = [...regions].sort((a, b) => a.timeSec - b.timeSec)
  const groups: Array<{ timeSec: number; items: Region[] }> = []
  for (const region of sorted) {
    const last = groups[groups.length - 1]
    if (last && Math.abs(region.timeSec - last.timeSec) <= TIME_EPS) {
      last.items.push(region)
    } else {
      groups.push({ timeSec: region.timeSec, items: [region] })
    }
  }
  return groups
}

export default function App(): JSX.Element {
  const [sidecar, setSidecar] = useState<SidecarState>({
    status: 'starting',
    port: null,
    error: null,
    python: null
  })
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [inputPath, setInputPath] = useState<string | null>(null)
  const [mediaUrl, setMediaUrl] = useState<string | null>(null)
  const [resultPath, setResultPath] = useState<string | null>(null)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [regions, setRegions] = useState<Region[]>([])
  const [engine, setEngine] = useState<'auto' | 'propainter' | 'delogo'>('auto')
  const [maxEdge, setMaxEdge] = useState(1280)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState('等待视频')
  const [error, setError] = useState<string | null>(null)
  const [compare, setCompare] = useState(false)
  const [slider, setSlider] = useState(52)
  const [usedEngine, setUsedEngine] = useState<string | null>(null)
  const [partial, setPartial] = useState(false)
  const [seekTo, setSeekTo] = useState<{ time: number; nonce: number } | null>(null)
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const jobIdRef = useRef<string | null>(null)

  const apiBase = sidecar.port ? `http://127.0.0.1:${sidecar.port}` : null

  useEffect(() => {
    if (!window.api) {
      setSidecar({
        status: 'error',
        port: null,
        error: '请通过 Electron 桌面窗口打开（npm run dev），浏览器里无法调用本地文件对话框。',
        python: null
      })
      return
    }
    void window.api.getSidecarState().then(setSidecar)
    return window.api.onSidecarState(setSidecar)
  }, [])

  useEffect(() => {
    if (!apiBase) return
    const load = async (): Promise<void> => {
      try {
        const response = await fetch(`${apiBase}/health`)
        if (!response.ok) return
        setHealth((await response.json()) as HealthInfo)
      } catch {
        setHealth(null)
      }
    }
    void load()
    const timer = window.setInterval(() => void load(), 4000)
    return () => window.clearInterval(timer)
  }, [apiBase])

  const loadVideo = useCallback(async (filePath: string | null) => {
    if (!filePath) return
    const url = await window.api.toMediaUrl(filePath)
    setInputPath(filePath)
    setMediaUrl(url)
    setRegions([])
    setResultPath(null)
    setResultUrl(null)
    setCompare(false)
    setPartial(false)
    setError(null)
    setProgress(0)
    setMessage('拖进度条到水印出现的时刻再框选，可增删改查不同时间的区域')
  }, [])

  const onDrop = useCallback(
    async (event: React.DragEvent) => {
      event.preventDefault()
      const file = event.dataTransfer.files[0]
      if (!file || !window.api) return
      const lower = file.name.toLowerCase()
      if (!VIDEO_EXT.some((ext) => lower.endsWith(ext))) {
        setError('请拖入视频文件')
        return
      }
      await loadVideo(window.api.getPathForFile(file))
    },
    [loadVideo]
  )

  const addRegion = useCallback((rect: Rect, timeSec: number) => {
    setRegions((current) => [
      ...current,
      { id: `${Date.now()}-${current.length}`, rect, timeSec }
    ])
  }, [])

  const updateRegion = useCallback((id: string, rect: Rect) => {
    setRegions((current) => current.map((region) => (region.id === id ? { ...region, rect } : region)))
  }, [])

  const deleteRegion = useCallback((id: string) => {
    setRegions((current) => current.filter((region) => region.id !== id))
  }, [])

  const deleteTimeGroup = useCallback((timeSec: number) => {
    setRegions((current) => current.filter((region) => Math.abs(region.timeSec - timeSec) > TIME_EPS))
  }, [])

  const timeGroups = useMemo(() => groupByTime(regions), [regions])

  const pixelRegions = useCallback(async (): Promise<Array<Rect & { timeSec: number }> | null> => {
    if (!regions.length || !apiBase || !inputPath) return null
    try {
      const response = await fetch(`${apiBase}/probe?path=${encodeURIComponent(inputPath)}`)
      if (!response.ok) return null
      const info = (await response.json()) as { width: number; height: number }
      return regions.map((region) => ({
        x: region.rect.x * info.width,
        y: region.rect.y * info.height,
        width: region.rect.width * info.width,
        height: region.rect.height * info.height,
        rotation: region.rect.rotation ?? 0,
        timeSec: region.timeSec
      }))
    } catch {
      return null
    }
  }, [apiBase, inputPath, regions])

  const startJob = useCallback(async () => {
    if (!apiBase || !inputPath || !regions.length) return
    const mapped = await pixelRegions()
    if (!mapped) {
      setError('无法读取视频尺寸')
      return
    }
    setBusy(true)
    setError(null)
    setCompare(false)
    setPartial(false)
    setProgress(0.04)
    setMessage('创建任务…')
    try {
      const created = await fetch(`${apiBase}/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          inputPath,
          regions: mapped,
          engine,
          maxEdge,
          track: true
        })
      })
      if (!created.ok) {
        throw new Error(await created.text())
      }
      const { jobId } = (await created.json()) as { jobId: string }
      jobIdRef.current = jobId
      setActiveJobId(jobId)
      const source = new EventSource(`${apiBase}/jobs/${jobId}/events`)
      await new Promise<void>((resolve, reject) => {
        let settled = false
        const finish = (fn: () => void): void => {
          if (settled) return
          settled = true
          source.close()
          fn()
        }
        source.onmessage = (event) => {
          const payload = JSON.parse(event.data) as JobEvent
          if (typeof payload.progress === 'number') setProgress(payload.progress)
          if (payload.message) setMessage(payload.message)
          if (payload.stage === 'done' && payload.outputPath) {
            setResultPath(payload.outputPath)
            setUsedEngine(payload.engine ?? engine)
            setPartial(Boolean(payload.partial))
            finish(() => resolve())
          }
          if (payload.stage === 'error') {
            finish(() => reject(new Error(payload.message || '处理失败')))
          }
        }
        source.onerror = () => {
          finish(() => reject(new Error('进度连接中断')))
        }
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setMessage('处理失败')
    } finally {
      jobIdRef.current = null
      setActiveJobId(null)
      setBusy(false)
    }
  }, [apiBase, engine, inputPath, maxEdge, pixelRegions, regions.length])

  const stopJob = useCallback(async () => {
    if (!apiBase || !jobIdRef.current) return
    setMessage('正在停止并生成已处理片段…')
    await fetch(`${apiBase}/jobs/${jobIdRef.current}/cancel`, { method: 'POST' })
  }, [apiBase])

  useEffect(() => {
    if (!resultPath) return
    void window.api.toMediaUrl(resultPath).then((url) => {
      setResultUrl(url)
      setCompare(true)
      setProgress(1)
      setMessage(partial ? '部分完成，可对比查看已处理片段' : '修复完成，可导出或对比查看')
    })
  }, [partial, resultPath])

  const exportResult = useCallback(async () => {
    if (!resultPath || !inputPath) return
    const target = await window.api.saveVideo(`${stem(inputPath)}-repaired.mp4`)
    if (!target) return
    await window.api.copyFile(resultPath, target)
    setMessage(`已导出到 ${basename(target)}`)
  }, [inputPath, resultPath])

  const installModel = useCallback(async () => {
    if (!apiBase) return
    setBusy(true)
    setMessage('正在下载 ProPainter 源码…')
    try {
      const response = await fetch(`${apiBase}/setup/propainter`, { method: 'POST' })
      if (!response.ok) throw new Error(await response.text())
      const healthRes = await fetch(`${apiBase}/health`)
      setHealth((await healthRes.json()) as HealthInfo)
      setMessage('ProPainter 源码已就绪。请再安装 CUDA 版 PyTorch 与 requirements-ai.txt')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [apiBase])

  const backendLabel = useMemo(() => {
    if (sidecar.status === 'ready') return '后端已连接'
    if (sidecar.status === 'error') return '后端启动失败'
    if (sidecar.status === 'stopped') return '后端已停止'
    return '正在启动后端'
  }, [sidecar.status])

  const canStart = Boolean(apiBase && inputPath && regions.length && !busy)

  return (
    <div
      className="flex h-full flex-col bg-[#09090b] text-zinc-100"
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => void onDrop(event)}
    >
      <header className="flex items-center justify-between border-b border-white/8 px-6 py-4">
        <div>
          <div className="text-[11px] tracking-[0.28em] text-amber-300/80">LOCAL INPAINTING</div>
          <h1 className="mt-1 text-xl font-medium tracking-tight">WaterMark 视频区域修复</h1>
        </div>
        <div className="flex items-center gap-4 text-xs text-zinc-400">
          <StatusDot ok={sidecar.status === 'ready'} label={backendLabel} />
          <StatusDot ok={Boolean(health?.ffmpegReady)} label="FFmpeg" />
          <StatusDot ok={Boolean(health?.cuda.cuda)} label={health?.cuda.device || 'CUDA'} />
          <StatusDot ok={Boolean(health?.propainterReady)} label="ProPainter" />
        </div>
      </header>

      <main className="grid min-h-0 flex-1 grid-cols-[1fr_320px] gap-5 p-5">
        <section className="min-h-0">
          {compare && mediaUrl && resultUrl ? (
            <CompareSlider before={mediaUrl} after={resultUrl} value={slider} onChange={setSlider} />
          ) : (
            <VideoCanvas
              src={mediaUrl}
              regions={regions}
              seekTo={seekTo}
              onAddRegion={addRegion}
              onUpdateRegion={updateRegion}
              onDeleteRegion={deleteRegion}
            />
          )}
        </section>

        <aside className="flex min-h-0 flex-col gap-4 overflow-auto rounded-2xl border border-white/8 bg-[#121216] p-4">
          <div className="rounded-xl border border-dashed border-white/15 bg-black/20 p-4">
            <div className="text-xs text-zinc-500">当前视频</div>
            <div className="mt-1 truncate text-sm">{inputPath ? basename(inputPath) : '尚未选择'}</div>
            <button
              className="mt-3 w-full rounded-lg bg-white/8 px-3 py-2 text-sm hover:bg-white/12"
              onClick={() => void window.api?.openVideo().then((path) => void loadVideo(path ?? null))}
              disabled={busy || !window.api}
            >
              选择视频
            </button>
          </div>

          <label className="block text-sm">
            <span className="text-xs text-zinc-500">修复引擎</span>
            <select
              className="mt-1 w-full rounded-lg border border-white/10 bg-[#1a1a20] px-3 py-2"
              value={engine}
              onChange={(event) => setEngine(event.target.value as typeof engine)}
              disabled={busy}
            >
              <option value="auto">自动（有模型用 AI，否则 FFmpeg）</option>
              <option value="propainter">ProPainter AI</option>
              <option value="delogo">FFmpeg 邻域修补</option>
            </select>
          </label>

          <label className="block text-sm">
            <span className="text-xs text-zinc-500">AI 处理最长边</span>
            <select
              className="mt-1 w-full rounded-lg border border-white/10 bg-[#1a1a20] px-3 py-2"
              value={maxEdge}
              onChange={(event) => setMaxEdge(Number(event.target.value))}
              disabled={busy}
            >
              <option value={720}>720（更省显存）</option>
              <option value={1280}>1280（推荐）</option>
              <option value={1920}>1920</option>
              <option value={0}>原始分辨率</option>
            </select>
          </label>

          <div className="rounded-xl border border-white/8 bg-black/20 p-3">
            <div className="mb-2 flex items-center justify-between text-xs text-zinc-500">
              <span>已框选 {regions.length} 块</span>
              <button
                className="text-zinc-400 hover:text-zinc-100"
                disabled={busy || regions.length === 0}
                onClick={() => setRegions([])}
              >
                清空
              </button>
            </div>
            {regions.length === 0 ? (
              <p className="text-xs leading-5 text-zinc-500">
                拖进度条到水印出现的时刻再框。同一时刻可框多块；点选后可拖动、缩放、旋转或删除。
              </p>
            ) : (
              <ul className="space-y-2">
                {timeGroups.map((group) => (
                  <li key={group.timeSec} className="rounded-lg bg-white/5 p-2">
                    <div className="mb-1 flex items-center justify-between gap-2">
                      <button
                        className="text-left text-xs text-amber-200 hover:text-amber-100"
                        disabled={busy}
                        onClick={() => setSeekTo({ time: group.timeSec, nonce: Date.now() })}
                      >
                        {group.timeSec.toFixed(1)}s · {group.items.length} 块
                      </button>
                      <button
                        className="text-[11px] text-zinc-500 hover:text-red-300"
                        disabled={busy}
                        onClick={() => deleteTimeGroup(group.timeSec)}
                      >
                        删除时刻
                      </button>
                    </div>
                    <ul className="space-y-1">
                      {group.items.map((region) => {
                        const index = regions.findIndex((item) => item.id === region.id)
                        return (
                          <li key={region.id} className="flex items-center justify-between text-xs">
                            <button
                              className="text-left text-zinc-200 hover:text-amber-200"
                              disabled={busy}
                              onClick={() => setSeekTo({ time: region.timeSec, nonce: Date.now() })}
                            >
                              区域 {index + 1}
                            </button>
                            <button
                              className="text-zinc-400 hover:text-red-300"
                              disabled={busy}
                              onClick={() => deleteRegion(region.id)}
                            >
                              删除
                            </button>
                          </li>
                        )
                      })}
                    </ul>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <p className="text-xs leading-5 text-zinc-500">
            底部进度条可拖动。画面只显示当前时刻附近的框，点选后可改位置和大小。请只处理你拥有或已获授权的素材。
          </p>

          {!health?.propainterReady ? (
            <button
              className="rounded-lg border border-amber-300/30 px-3 py-2 text-sm text-amber-200 hover:bg-amber-300/10"
              onClick={() => void installModel()}
              disabled={busy || !apiBase}
            >
              下载 ProPainter 源码
            </button>
          ) : null}

          <button
            className={`mt-auto rounded-xl px-3 py-3 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-40 ${
              busy ? 'bg-zinc-100 text-zinc-950' : 'bg-amber-300 text-zinc-950'
            }`}
            disabled={busy ? !activeJobId : !canStart}
            onClick={() => void (busy ? stopJob() : startJob())}
          >
            {busy ? '停止' : '开始修复'}
          </button>

          {resultPath ? (
            <div className="grid grid-cols-2 gap-2">
              <button
                className="rounded-lg bg-white/8 px-3 py-2 text-sm"
                onClick={() => setCompare((value) => !value)}
              >
                {compare ? '返回框选' : '左右对比'}
              </button>
              <button className="rounded-lg bg-white/8 px-3 py-2 text-sm" onClick={() => void exportResult()}>
                导出
              </button>
              {partial ? (
                <button
                  className="col-span-2 rounded-lg bg-amber-300/90 px-3 py-2 text-sm font-medium text-zinc-950"
                  disabled={busy}
                  onClick={() => {
                    setCompare(false)
                    void startJob()
                  }}
                >
                  继续处理全片
                </button>
              ) : null}
              <button
                className="col-span-2 rounded-lg bg-white/8 px-3 py-2 text-sm"
                onClick={() => void window.api?.revealInFolder(resultPath)}
              >
                打开所在文件夹
              </button>
            </div>
          ) : null}
        </aside>
      </main>

      <footer className="border-t border-white/8 px-6 py-3">
        <div className="mb-2 flex items-center justify-between text-xs text-zinc-400">
          <span>{message}</span>
          <span>
            {usedEngine ? `引擎：${usedEngine}` : ''} {partial ? '部分 ' : ''}
            {Math.round(progress * 100)}%
          </span>
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-white/8">
          <div className="h-full bg-amber-300 transition-all" style={{ width: `${Math.round(progress * 100)}%` }} />
        </div>
        {error || sidecar.error ? (
          <pre className="mt-2 max-h-24 overflow-auto whitespace-pre-wrap text-[11px] text-red-300">
            {error || sidecar.error}
          </pre>
        ) : null}
      </footer>
    </div>
  )
}

function StatusDot({ ok, label }: { ok: boolean; label: string }): JSX.Element {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={`h-1.5 w-1.5 rounded-full ${ok ? 'bg-emerald-400' : 'bg-zinc-600'}`} />
      {label}
    </span>
  )
}
