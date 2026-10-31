export function fmtDuration(ms: number | null | undefined): string {
  if (ms == null) return ''
  const s = Math.max(0, Math.round(ms / 1000))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const rm = s % 60
  if (s < 3600) return `${m}m ${String(rm).padStart(2, '0')}s`
  const h = Math.floor(s / 3600)
  const rs = s % 3600
  return `${h}h ${String(Math.floor(rs / 60)).padStart(2, '0')}m ${String(rs % 60).padStart(2, '0')}s`
}

export function fmtTokens(n: number | null | undefined): string {
  if (n == null || n <= 0) return ''
  if (n < 1000) return String(n)
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`
  return `${(n / 1_000_000).toFixed(1)}M`
}
