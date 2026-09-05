import type { JSX } from 'react'

type Props = {
  before: string
  after: string
  value: number
  onChange: (value: number) => void
}

export function CompareSlider({ before, after, value, onChange }: Props): JSX.Element {
  return (
    <div className="relative h-full w-full overflow-hidden rounded-2xl bg-black">
      <video src={before} className="absolute inset-0 h-full w-full object-contain" muted loop autoPlay />
      <video
        src={after}
        className="absolute inset-0 h-full w-full object-contain"
        muted
        loop
        autoPlay
        style={{ clipPath: `inset(0 ${100 - value}% 0 0)` }}
      />
      <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-amber-200" style={{ left: `${value}%` }} />
      <input
        type="range"
        min={0}
        max={100}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="absolute inset-x-6 bottom-4 h-8 cursor-ew-resize accent-amber-300"
      />
      <div className="pointer-events-none absolute left-4 top-4 rounded-full bg-black/60 px-2 py-1 text-[11px] text-zinc-200">
        原片
      </div>
      <div className="pointer-events-none absolute right-4 top-4 rounded-full bg-amber-300/90 px-2 py-1 text-[11px] text-zinc-900">
        结果
      </div>
    </div>
  )
}
