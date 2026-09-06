import { useCallback, useEffect, useRef, useState, type JSX, type PointerEvent as ReactPointerEvent } from 'react'

type Props = {
  before: string
  after: string
  value: number
  onChange: (value: number) => void
}

function formatTime(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '0:00.0'
  const minutes = Math.floor(sec / 60)
  const seconds = sec - minutes * 60
  return `${minutes}:${seconds.toFixed(1).padStart(4, '0')}`
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

export function CompareSlider({ before, after, value, onChange }: Props): JSX.Element {
  const stageRef = useRef<HTMLDivElement>(null)
  const seekBarRef = useRef<HTMLDivElement>(null)
  const beforeRef = useRef<HTMLVideoElement>(null)
  const afterRef = useRef<HTMLVideoElement>(null)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [scrubTime, setScrubTime] = useState<number | null>(null)
  const scrubbingRef = useRef(false)
  const draggingLineRef = useRef(false)

  const syncAfter = useCallback((time: number): void => {
    const clip = afterRef.current
    if (!clip) return
    const limit = clip.duration || time
    const next = clamp(time, 0, limit)
    if (Math.abs(clip.currentTime - next) > 0.08) clip.currentTime = next
  }, [])

  const seekBoth = useCallback(
    (time: number): void => {
      const source = beforeRef.current
      if (!source) return
      const next = clamp(time, 0, source.duration || time)
      source.currentTime = next
      syncAfter(next)
      setCurrentTime(next)
    },
    [syncAfter]
  )

  const togglePlay = useCallback((): void => {
    const source = beforeRef.current
    const clip = afterRef.current
    if (!source) return
    if (source.paused) {
      void source.play()
      void clip?.play()
      setPlaying(true)
    } else {
      source.pause()
      clip?.pause()
      setPlaying(false)
    }
  }, [])

  const ratioFromPointer = useCallback((clientX: number): number => {
    const el = stageRef.current
    if (!el) return value
    const box = el.getBoundingClientRect()
    if (box.width <= 0) return value
    return clamp(((clientX - box.left) / box.width) * 100, 0, 100)
  }, [value])

  const beginLineDrag = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>): void => {
      event.preventDefault()
      event.stopPropagation()
      draggingLineRef.current = true
      event.currentTarget.setPointerCapture(event.pointerId)
      onChange(ratioFromPointer(event.clientX))
    },
    [onChange, ratioFromPointer]
  )

  const moveLineDrag = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>): void => {
      if (!draggingLineRef.current) return
      onChange(ratioFromPointer(event.clientX))
    },
    [onChange, ratioFromPointer]
  )

  const endLineDrag = useCallback((event: ReactPointerEvent<HTMLDivElement>): void => {
    draggingLineRef.current = false
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }, [])

  const timeFromPointer = useCallback(
    (clientX: number): number => {
      const el = seekBarRef.current
      if (!el || duration <= 0) return 0
      const box = el.getBoundingClientRect()
      if (box.width <= 0) return 0
      return clamp((clientX - box.left) / box.width, 0, 1) * duration
    },
    [duration]
  )

  const beginScrub = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>): void => {
      event.preventDefault()
      scrubbingRef.current = true
      event.currentTarget.setPointerCapture(event.pointerId)
      const time = timeFromPointer(event.clientX)
      setScrubTime(time)
      seekBoth(time)
      beforeRef.current?.pause()
      afterRef.current?.pause()
      setPlaying(false)
    },
    [seekBoth, timeFromPointer]
  )

  const moveScrub = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>): void => {
      if (!scrubbingRef.current) return
      const time = timeFromPointer(event.clientX)
      setScrubTime(time)
      seekBoth(time)
    },
    [seekBoth, timeFromPointer]
  )

  const endScrub = useCallback((event: ReactPointerEvent<HTMLDivElement>): void => {
    scrubbingRef.current = false
    setScrubTime(null)
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }, [])

  useEffect(() => {
    setPlaying(false)
    setCurrentTime(0)
    setDuration(0)
    setScrubTime(null)
    scrubbingRef.current = false
  }, [before, after])

  const displayTime = scrubTime ?? currentTime
  const progress = duration > 0 ? displayTime / duration : 0

  return (
    <div className="flex h-full w-full flex-col overflow-hidden rounded-2xl bg-black">
      <div ref={stageRef} className="relative min-h-0 flex-1 overflow-hidden">
        <video
          ref={beforeRef}
          src={before}
          className="absolute inset-0 h-full w-full object-contain"
          preload="auto"
          onLoadedMetadata={(event) => setDuration(event.currentTarget.duration || 0)}
          onDurationChange={(event) => setDuration(event.currentTarget.duration || 0)}
          onTimeUpdate={(event) => {
            if (scrubbingRef.current) return
            const time = event.currentTarget.currentTime
            setCurrentTime(time)
            syncAfter(time)
          }}
          onPlay={() => {
            setPlaying(true)
            void afterRef.current?.play()
          }}
          onPause={() => {
            if (scrubbingRef.current) return
            setPlaying(false)
            afterRef.current?.pause()
          }}
          onEnded={() => {
            setPlaying(false)
            afterRef.current?.pause()
          }}
        />
        <video
          ref={afterRef}
          src={after}
          muted
          className="absolute inset-0 h-full w-full object-contain"
          preload="auto"
          style={{ clipPath: `inset(0 0 0 ${value}%)` }}
        />
        <div
          className="absolute inset-y-0 z-10 w-6 -translate-x-1/2 cursor-ew-resize touch-none"
          style={{ left: `${value}%` }}
          onPointerDown={beginLineDrag}
          onPointerMove={moveLineDrag}
          onPointerUp={endLineDrag}
          onPointerCancel={endLineDrag}
          role="slider"
          aria-label="对比位置"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(value)}
        >
          <div className="pointer-events-none absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 bg-amber-200" />
          <div className="pointer-events-none absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 items-center gap-0.5 rounded-full border-2 border-zinc-950 bg-amber-300 px-1 py-1 shadow-[0_0_0_3px_rgba(251,191,36,0.28)]">
            <svg viewBox="0 0 8 10" className="h-2.5 w-2 text-zinc-950" fill="currentColor" aria-hidden>
              <path d="M7 1 1 5l6 4V1z" />
            </svg>
            <svg viewBox="0 0 8 10" className="h-2.5 w-2 text-zinc-950" fill="currentColor" aria-hidden>
              <path d="M1 1l6 4-6 4V1z" />
            </svg>
          </div>
        </div>
        <div className="pointer-events-none absolute left-4 top-4 rounded-full bg-black/60 px-2 py-1 text-[11px] text-zinc-200">
          原片
        </div>
        <div className="pointer-events-none absolute right-4 top-4 rounded-full bg-amber-300/90 px-2 py-1 text-[11px] text-zinc-900">
          结果
        </div>
      </div>

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
        </div>
        <span className="w-[4.8rem] shrink-0 font-mono text-[11px] tabular-nums text-zinc-500">
          {formatTime(duration)}
        </span>
      </div>
    </div>
  )
}
