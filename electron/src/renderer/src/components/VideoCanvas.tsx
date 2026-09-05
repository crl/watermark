import { useCallback, useEffect, useMemo, useRef, useState, type JSX, type PointerEvent as ReactPointerEvent } from 'react'
import type { Rect } from '@shared/types'

export type Region = {
  id: string
  rect: Rect
  timeSec: number
}

export const TIME_EPS = 0.15

type Props = {
  src: string | null
  regions: Region[]
  seekTo: { time: number; nonce: number } | null
  jobProgress?: number
  jobMessage?: string
  jobStage?: string
  busy?: boolean
  paused?: boolean
  jobDone?: boolean
  jobPartial?: boolean
  onAddRegion: (rect: Rect, timeSec: number) => void
  onUpdateRegion: (id: string, rect: Rect) => void
  onDeleteRegion: (id: string) => void
}

type Handle = 'nw' | 'ne' | 'sw' | 'se' | 'n' | 'e' | 's' | 'w'

type Edit =
  | { kind: 'draw'; startX: number; startY: number }
  | { kind: 'move'; id: string; origin: Rect; startNX: number; startNY: number }
  | { kind: 'resize'; id: string; handle: Handle; origin: Rect }
  | { kind: 'rotate'; id: string; origin: Rect; startAngle: number }

type Frame = {
  left: number
  top: number
  width: number
  height: number
}

type Point = { x: number; y: number }

const COLORS = ['#e4b56a', '#7dd3fc', '#f9a8d4', '#86efac', '#fdba74']
const MIN_SIZE = 0.005
const CORNERS: Handle[] = ['nw', 'ne', 'sw', 'se']
const EDGES: Handle[] = ['n', 'e', 's', 'w']

function formatTime(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '0:00.0'
  const minutes = Math.floor(sec / 60)
  const seconds = sec - minutes * 60
  return `${minutes}:${seconds.toFixed(1).padStart(4, '0')}`
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function timeFromClientX(clientX: number, el: HTMLElement, duration: number): number {
  const box = el.getBoundingClientRect()
  if (box.width <= 0 || duration <= 0) return 0
  return clamp((clientX - box.left) / box.width, 0, 1) * duration
}

function rotationOf(rect: Rect): number {
  return rect.rotation ?? 0
}

function rectCenter(rect: Rect): Point {
  return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 }
}

function toLocalPoint(point: Point, rect: Rect): Point {
  const center = rectCenter(rect)
  const dx = point.x - center.x
  const dy = point.y - center.y
  const cos = Math.cos(-rotationOf(rect))
  const sin = Math.sin(-rotationOf(rect))
  return {
    x: center.x + dx * cos - dy * sin,
    y: center.y + dx * sin + dy * cos
  }
}

function pointerAngle(point: Point, rect: Rect): number {
  const center = rectCenter(rect)
  return Math.atan2(point.y - center.y, point.x - center.x)
}

function translateRect(origin: Rect, dx: number, dy: number): Rect {
  return {
    x: clamp(origin.x + dx, 0, 1 - origin.width),
    y: clamp(origin.y + dy, 0, 1 - origin.height),
    width: origin.width,
    height: origin.height,
    rotation: rotationOf(origin)
  }
}

function resizeRect(origin: Rect, handle: Handle, point: Point): Rect {
  const local = toLocalPoint(point, origin)
  let x1 = origin.x
  let y1 = origin.y
  let x2 = origin.x + origin.width
  let y2 = origin.y + origin.height
  if (handle === 'nw' || handle === 'sw' || handle === 'w') x1 = local.x
  if (handle === 'ne' || handle === 'se' || handle === 'e') x2 = local.x
  if (handle === 'nw' || handle === 'ne' || handle === 'n') y1 = local.y
  if (handle === 'sw' || handle === 'se' || handle === 's') y2 = local.y
  const x = clamp(Math.min(x1, x2), 0, 1 - MIN_SIZE)
  const y = clamp(Math.min(y1, y2), 0, 1 - MIN_SIZE)
  return {
    x,
    y,
    width: clamp(Math.abs(x2 - x1), MIN_SIZE, 1 - x),
    height: clamp(Math.abs(y2 - y1), MIN_SIZE, 1 - y),
    rotation: rotationOf(origin)
  }
}

function rotateRect(origin: Rect, startAngle: number, point: Point): Rect {
  return {
    ...origin,
    rotation: rotationOf(origin) + (pointerAngle(point, origin) - startAngle)
  }
}

function cornerClass(handle: Handle): string {
  const pos: Record<Handle, string> = {
    nw: 'left-0 top-0 -translate-x-1/2 -translate-y-1/2 cursor-nwse-resize',
    ne: 'right-0 top-0 translate-x-1/2 -translate-y-1/2 cursor-nesw-resize',
    sw: 'bottom-0 left-0 -translate-x-1/2 translate-y-1/2 cursor-nesw-resize',
    se: 'bottom-0 right-0 translate-x-1/2 translate-y-1/2 cursor-nwse-resize',
    n: 'left-2 right-2 top-0 h-2 -translate-y-1/2 cursor-ns-resize',
    s: 'left-2 right-2 bottom-0 h-2 translate-y-1/2 cursor-ns-resize',
    e: 'top-2 bottom-2 right-0 w-2 translate-x-1/2 cursor-ew-resize',
    w: 'top-2 bottom-2 left-0 w-2 -translate-x-1/2 cursor-ew-resize'
  }
  return pos[handle]
}

export function VideoCanvas({
  src,
  regions,
  seekTo,
  jobProgress = 0,
  jobMessage = '',
  jobStage = '',
  busy = false,
  paused = false,
  jobDone = false,
  jobPartial = false,
  onAddRegion,
  onUpdateRegion,
  onDeleteRegion
}: Props): JSX.Element {
  const wrapRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const seekBarRef = useRef<HTMLDivElement>(null)
  const [edit, setEdit] = useState<Edit | null>(null)
  const [preview, setPreview] = useState<Rect | null>(null)
  const [liveRect, setLiveRect] = useState<Rect | null>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [scrubTime, setScrubTime] = useState<number | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [frame, setFrame] = useState<Frame>({ left: 0, top: 0, width: 0, height: 0 })
  const scrubbingRef = useRef(false)
  const scrubTimeRef = useRef(0)
  const seekRafRef = useRef(0)
  const pendingSeekRef = useRef<number | null>(null)

  const updateFrame = useCallback((): void => {
    const video = videoRef.current
    const wrap = wrapRef.current
    if (!video || !wrap || !video.videoWidth) return
    const scale = Math.min(wrap.clientWidth / video.videoWidth, wrap.clientHeight / video.videoHeight)
    const width = video.videoWidth * scale
    const height = video.videoHeight * scale
    setFrame({
      left: (wrap.clientWidth - width) / 2,
      top: (wrap.clientHeight - height) / 2,
      width,
      height
    })
  }, [])

  useEffect(() => {
    const wrap = wrapRef.current
    if (!wrap) return
    const observer = new ResizeObserver(() => updateFrame())
    observer.observe(wrap)
    return () => observer.disconnect()
  }, [updateFrame, src])

  useEffect(() => {
    setPlaying(false)
    setCurrentTime(0)
    setDuration(0)
    setScrubTime(null)
    scrubbingRef.current = false
    setSelectedId(null)
    setEdit(null)
    setPreview(null)
    setLiveRect(null)
  }, [src])

  useEffect(() => {
    if (!seekTo || !videoRef.current) return
    videoRef.current.pause()
    setPlaying(false)
    pendingSeekRef.current = seekTo.time
    videoRef.current.currentTime = seekTo.time
    setCurrentTime(seekTo.time)
  }, [seekTo])

  const clientToNorm = useCallback((clientX: number, clientY: number): Point | null => {
    const wrap = wrapRef.current
    const video = videoRef.current
    if (!wrap || !video || !video.videoWidth) return null
    const box = wrap.getBoundingClientRect()
    const scale = Math.min(box.width / video.videoWidth, box.height / video.videoHeight)
    const dw = video.videoWidth * scale
    const dh = video.videoHeight * scale
    const left = box.left + (box.width - dw) / 2
    const top = box.top + (box.height - dh) / 2
    return {
      x: clamp((clientX - left) / dw, 0, 1),
      y: clamp((clientY - top) / dh, 0, 1)
    }
  }, [])

  const toVideoRect = useCallback(
    (clientX: number, clientY: number, origin: { startX: number; startY: number }): Rect | null => {
      const point = clientToNorm(clientX, clientY)
      const start = clientToNorm(origin.startX, origin.startY)
      if (!point || !start) return null
      return {
        x: Math.min(start.x, point.x),
        y: Math.min(start.y, point.y),
        width: Math.abs(point.x - start.x),
        height: Math.abs(point.y - start.y),
        rotation: 0
      }
    },
    [clientToNorm]
  )

  const pauseVideo = useCallback((): void => {
    videoRef.current?.pause()
    setPlaying(false)
  }, [])

  const seek = useCallback((time: number): void => {
    const video = videoRef.current
    if (!video || !Number.isFinite(time)) return
    const next = clamp(time, 0, video.duration || time)
    pendingSeekRef.current = next
    video.currentTime = next
    setCurrentTime(next)
  }, [])

  const previewSeek = useCallback((time: number): void => {
    if (seekRafRef.current) cancelAnimationFrame(seekRafRef.current)
    seekRafRef.current = requestAnimationFrame(() => {
      seekRafRef.current = 0
      const video = videoRef.current
      if (!video) return
      video.currentTime = clamp(time, 0, video.duration || time)
    })
  }, [])

  const timeFromPointer = useCallback((clientX: number): number => {
    const el = seekBarRef.current
    if (!el) return 0
    const dur = videoRef.current?.duration || duration
    return timeFromClientX(clientX, el, dur)
  }, [duration])

  const onScrub = useCallback(
    (time: number): void => {
      if (!scrubbingRef.current) return
      scrubTimeRef.current = time
      setScrubTime(time)
      previewSeek(time)
    },
    [previewSeek]
  )

  const beginScrub = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>): void => {
      if (!duration || event.button !== 0) return
      event.preventDefault()
      event.currentTarget.setPointerCapture(event.pointerId)
      const time = timeFromPointer(event.clientX)
      scrubbingRef.current = true
      scrubTimeRef.current = time
      pendingSeekRef.current = time
      setScrubTime(time)
      pauseVideo()
      previewSeek(time)
    },
    [duration, pauseVideo, previewSeek, timeFromPointer]
  )

  const moveScrub = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>): void => {
      if (!scrubbingRef.current) return
      onScrub(timeFromPointer(event.clientX))
    },
    [onScrub, timeFromPointer]
  )

  const endScrub = useCallback((): void => {
    if (!scrubbingRef.current) return
    scrubbingRef.current = false
    if (seekRafRef.current) {
      cancelAnimationFrame(seekRafRef.current)
      seekRafRef.current = 0
    }
    seek(scrubTimeRef.current)
    setScrubTime(null)
  }, [seek])

  useEffect(() => {
    return () => {
      if (seekRafRef.current) cancelAnimationFrame(seekRafRef.current)
    }
  }, [])

  const onVideoTime = useCallback((event: { currentTarget: HTMLVideoElement }): void => {
    if (scrubbingRef.current) return
    const time = event.currentTarget.currentTime
    const pending = pendingSeekRef.current
    if (pending != null && pending > 0.2 && time < 0.05) return
    if (pending != null && Math.abs(time - pending) <= 0.4) pendingSeekRef.current = null
    setCurrentTime(time)
  }, [])

  useEffect(() => {
    const onMove = (event: PointerEvent): void => {
      if (!edit) return
      if (edit.kind === 'draw') {
        setPreview(toVideoRect(event.clientX, event.clientY, edit))
        return
      }
      const point = clientToNorm(event.clientX, event.clientY)
      if (!point) return
      if (edit.kind === 'move') {
        setLiveRect(translateRect(edit.origin, point.x - edit.startNX, point.y - edit.startNY))
      } else if (edit.kind === 'resize') {
        setLiveRect(resizeRect(edit.origin, edit.handle, point))
      } else {
        setLiveRect(rotateRect(edit.origin, edit.startAngle, point))
      }
    }
    const onUp = (event: PointerEvent): void => {
      if (!edit) return
      if (edit.kind === 'draw') {
        const next = toVideoRect(event.clientX, event.clientY, edit)
        const timeSec = videoRef.current?.currentTime ?? 0
        setEdit(null)
        setPreview(null)
        if (next && next.width > MIN_SIZE && next.height > MIN_SIZE) {
          onAddRegion(next, timeSec)
        } else {
          setSelectedId(null)
        }
        return
      }
      const point = clientToNorm(event.clientX, event.clientY)
      const next =
        edit.kind === 'move' && point
          ? translateRect(edit.origin, point.x - edit.startNX, point.y - edit.startNY)
          : edit.kind === 'resize' && point
            ? resizeRect(edit.origin, edit.handle, point)
            : edit.kind === 'rotate' && point
              ? rotateRect(edit.origin, edit.startAngle, point)
              : null
      if (next) onUpdateRegion(edit.id, next)
      setEdit(null)
      setLiveRect(null)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
  }, [clientToNorm, edit, onAddRegion, onUpdateRegion, toVideoRect])

  const togglePlay = useCallback((): void => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) {
      void video.play()
      setPlaying(true)
    } else {
      video.pause()
      setPlaying(false)
    }
  }, [])

  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if (event.code !== 'Space') return
      const target = event.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'BUTTON')) return
      if (!src) return
      event.preventDefault()
      togglePlay()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [src, togglePlay])

  const displayTime = scrubTime ?? currentTime
  const visible = regions.filter((region) => Math.abs(region.timeSec - displayTime) <= TIME_EPS)
  const marks = useMemo(() => {
    const times = [...new Set(regions.map((region) => region.timeSec))]
    times.sort((a, b) => a - b)
    return times
  }, [regions])

  useEffect(() => {
    if (selectedId && !visible.some((region) => region.id === selectedId)) {
      setSelectedId(null)
    }
  }, [selectedId, visible])

  const startDraw = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (event.button !== 0) return
    event.preventDefault()
    pauseVideo()
    setSelectedId(null)
    setEdit({ kind: 'draw', startX: event.clientX, startY: event.clientY })
  }

  const startMove = (event: ReactPointerEvent<HTMLDivElement>, region: Region): void => {
    if (event.button !== 0) return
    event.preventDefault()
    event.stopPropagation()
    pauseVideo()
    setSelectedId(region.id)
    const point = clientToNorm(event.clientX, event.clientY)
    if (!point) return
    setEdit({ kind: 'move', id: region.id, origin: region.rect, startNX: point.x, startNY: point.y })
  }

  const startResize = (event: ReactPointerEvent<HTMLDivElement>, region: Region, handle: Handle): void => {
    if (event.button !== 0) return
    event.preventDefault()
    event.stopPropagation()
    pauseVideo()
    setSelectedId(region.id)
    setEdit({ kind: 'resize', id: region.id, handle, origin: region.rect })
  }

  const startRotate = (event: ReactPointerEvent<HTMLDivElement>, region: Region): void => {
    if (event.button !== 0) return
    event.preventDefault()
    event.stopPropagation()
    pauseVideo()
    setSelectedId(region.id)
    const point = clientToNorm(event.clientX, event.clientY)
    if (!point) return
    setEdit({ kind: 'rotate', id: region.id, origin: region.rect, startAngle: pointerAngle(point, region.rect) })
  }

  const progress = duration > 0 ? displayTime / duration : 0
  const jobFill = busy ? clamp(jobProgress, 0, 1) : jobDone || jobPartial ? 1 : 0
  const preparing = !jobStage || ['probe', 'extract', 'track', 'mask'].includes(jobStage)
  const jobLabel = busy
    ? paused
      ? `已暂停  ${Math.round(jobFill * 100)}%`
      : `${preparing ? '准备中' : '修复中'} · ${jobMessage || '处理中'}  ${Math.round(jobFill * 100)}%`
    : jobPartial
      ? '部分完成'
      : jobDone
        ? '修复完成'
        : '未开始处理'

  return (
    <div className="flex h-full w-full flex-col overflow-hidden rounded-2xl bg-black">
      <div ref={wrapRef} className="relative min-h-0 flex-1">
        {src ? (
          <video
            ref={videoRef}
            src={src}
            preload="auto"
            className="relative z-0 h-full w-full object-contain"
            onLoadedMetadata={(event) => {
              setDuration(event.currentTarget.duration || 0)
              updateFrame()
            }}
            onDurationChange={(event) => setDuration(event.currentTarget.duration || 0)}
            onTimeUpdate={onVideoTime}
            onSeeked={onVideoTime}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onEnded={() => setPlaying(false)}
          />
        ) : (
          <div className="flex h-full items-center justify-center px-8 text-center text-sm text-zinc-500">
            用底部进度条拖到水印出现的时刻，再在画面上框选。同一时刻可框多块，也可改位置或删除。
          </div>
        )}
        {src ? (
          <div className="absolute inset-0 z-10 cursor-crosshair" onPointerDown={startDraw} />
        ) : null}
        {visible.map((region) => {
          const index = regions.findIndex((item) => item.id === region.id)
          const rect = selectedId === region.id && liveRect ? liveRect : region.rect
          const selected = selectedId === region.id
          return (
            <div
              key={region.id}
              className={`absolute z-20 border-2 bg-white/10 ${selected ? 'cursor-move' : 'cursor-pointer'}`}
              style={{
                borderColor: COLORS[index % COLORS.length],
                left: frame.left + rect.x * frame.width,
                top: frame.top + rect.y * frame.height,
                width: rect.width * frame.width,
                height: rect.height * frame.height,
                transform: `rotate(${rotationOf(rect)}rad)`,
                transformOrigin: 'center center',
                boxShadow: selected ? `0 0 0 1px ${COLORS[index % COLORS.length]}` : undefined
              }}
              onPointerDown={(event) => startMove(event, region)}
            >
              <div
                className="absolute -top-6 left-0 rounded px-1.5 py-0.5 text-[10px] font-medium text-zinc-950"
                style={{ background: COLORS[index % COLORS.length] }}
              >
                {region.timeSec.toFixed(1)}s · {index + 1}
              </div>
              <button
                type="button"
                className="absolute -left-2 -top-2 z-30 flex h-5 w-5 items-center justify-center rounded-full bg-zinc-950 text-[10px] text-zinc-100 hover:bg-red-500"
                title="删除此区域"
                onPointerDown={(event) => event.stopPropagation()}
                onClick={(event) => {
                  event.stopPropagation()
                  onDeleteRegion(region.id)
                  setSelectedId(null)
                }}
              >
                ×
              </button>
              {selected ? (
                <>
                  {EDGES.map((edge) => (
                    <div
                      key={edge}
                      className={`absolute z-20 ${cornerClass(edge)}`}
                      onPointerDown={(event) => startResize(event, region, edge)}
                    />
                  ))}
                  {CORNERS.map((corner) => (
                    <div
                      key={corner}
                      className={`absolute z-30 h-3 w-3 rounded-sm border border-zinc-950 bg-white ${cornerClass(corner)}`}
                      onPointerDown={(event) => startResize(event, region, corner)}
                    />
                  ))}
                  <div
                    className="absolute z-30 flex cursor-grab flex-col items-center active:cursor-grabbing"
                    style={{ right: 0, top: 0, transform: 'translate(50%, calc(-100% - 10px))' }}
                    title="旋转"
                    onPointerDown={(event) => startRotate(event, region)}
                  >
                    <div className="h-3.5 w-3.5 rounded-full border-2 border-zinc-950 bg-amber-300" />
                    <div className="h-2.5 w-px bg-amber-300" />
                  </div>
                </>
              ) : null}
            </div>
          )
        })}
        {preview && frame.width > 0 ? (
          <div
            className="pointer-events-none absolute z-20 border-2 border-white/80 bg-white/10"
            style={{
              left: frame.left + preview.x * frame.width,
              top: frame.top + preview.y * frame.height,
              width: preview.width * frame.width,
              height: preview.height * frame.height
            }}
          />
        ) : null}
      </div>

      {src ? (
        <div className="flex shrink-0 items-center gap-3 border-t border-white/10 bg-[#121216] px-3 py-2.5">
          <button
            type="button"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-amber-300 text-zinc-950 hover:bg-amber-200"
            onClick={togglePlay}
            title={playing ? '暂停' : '播放'}
          >
            {playing ? (
              <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="currentColor" aria-hidden>
                <rect x="3" y="2" width="4" height="12" rx="1" />
                <rect x="9" y="2" width="4" height="12" rx="1" />
              </svg>
            ) : (
              <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="currentColor" aria-hidden>
                <path d="M4 2.5v11l9-5.5-9-5.5z" />
              </svg>
            )}
          </button>
          <span className="w-[4.8rem] shrink-0 text-right font-mono text-[11px] tabular-nums text-zinc-300">
            {formatTime(displayTime)}
          </span>
          <div className="flex min-w-0 flex-1 flex-col justify-center">
            <div className="relative h-8">
              <div className="pointer-events-none absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-white/12">
                <div className="h-full rounded-full bg-amber-300" style={{ width: `${Math.min(100, progress * 100)}%` }} />
              </div>
              {duration > 0
                ? marks.map((time) => (
                    <div
                      key={`tick-${time}`}
                      className="pointer-events-none absolute top-1/2 z-0 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-sky-300"
                      style={{ left: `${(time / duration) * 100}%` }}
                    />
                  ))
                : null}
              <div
                ref={seekBarRef}
                role="slider"
                aria-label="播放进度"
                aria-valuemin={0}
                aria-valuemax={duration || 0}
                aria-valuenow={displayTime}
                className="absolute inset-0 z-10 cursor-pointer touch-none"
                onPointerDown={beginScrub}
                onPointerMove={moveScrub}
                onPointerUp={endScrub}
                onPointerCancel={endScrub}
              />
              <div
                className="pointer-events-none absolute top-1/2 z-20 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-zinc-950 bg-amber-300 shadow-[0_0_0_3px_rgba(251,191,36,0.28)]"
                style={{ left: `${Math.min(100, progress * 100)}%` }}
              />
            </div>
            {duration > 0 && marks.length > 0 ? (
              <div className="relative h-3">
                {marks.map((time) => (
                  <button
                    key={time}
                    type="button"
                    title={`跳到 ${formatTime(time)}`}
                    className="wm-mark absolute top-0 h-2.5 w-2.5 -translate-x-1/2 rounded-full bg-sky-300 hover:bg-amber-200"
                    style={{ left: `${(time / duration) * 100}%` }}
                    onClick={() => {
                      pauseVideo()
                      seek(time)
                    }}
                  />
                ))}
              </div>
            ) : null}
            <div className="pointer-events-none relative mt-1 h-5 overflow-hidden rounded-full bg-white/10">
              <div
                className="absolute inset-y-0 left-0 bg-amber-300/85 transition-all"
                style={{ width: `${Math.round(jobFill * 100)}%` }}
              />
              <div
                className={`relative z-10 truncate px-2 text-center text-[10px] leading-5 ${
                  jobFill > 0.35 ? 'text-zinc-950' : 'text-zinc-400'
                }`}
              >
                {jobLabel}
              </div>
            </div>
          </div>
          <span className="w-[4.8rem] shrink-0 font-mono text-[11px] tabular-nums text-zinc-500">
            {formatTime(duration)}
          </span>
        </div>
      ) : null}
    </div>
  )
}
