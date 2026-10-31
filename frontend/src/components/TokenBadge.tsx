import { fmtTokens } from '../utils/format'

export function TokenBadge({ used, window }: { used: number | null | undefined; window: number | null | undefined }) {
  if (!window || !used) return null
  const percent = Math.min(100, Math.round((used / window) * 100))
  return (
    <span
      className="font-num shrink-0 rounded-lg bg-zinc-100/80 px-2.5 py-1.5 text-[11px] text-zinc-600"
      title={`上下文占用：${fmtTokens(used)} / ${fmtTokens(window)} tokens（${percent}%）`}
    >
      <span className="hidden lg:inline">上下文 {fmtTokens(used)} / {fmtTokens(window)} · </span>{percent}%
    </span>
  )
}
