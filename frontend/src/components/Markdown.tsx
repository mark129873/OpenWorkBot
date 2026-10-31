import { useState } from 'react'
import { ArrowClockwise, Check, Copy } from '@phosphor-icons/react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'

function SafeImage({ src, alt }: { src?: string; alt?: string }) {
  const [loadedSrc, setLoadedSrc] = useState<string | null>(null)
  if (!src) return <span>{alt || '图片不可用'}</span>
  if (loadedSrc === src) return <img src={src} alt={alt || ''} referrerPolicy="no-referrer" />
  return <button type="button" onClick={() => setLoadedSrc(src)} className="rounded border border-zinc-300 px-2 py-1 text-sm underline">加载图片{alt ? `：${alt}` : ''}</button>
}

export default function MarkdownWithCopy({
  text,
  streaming = false,
  onRegenerate,
}: {
  text: string
  streaming?: boolean
  onRegenerate?: () => void
}) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    } catch {
      /* 剪贴板不可用时静默。 */
    }
  }

  return (
    <div className="group relative min-w-0">
      <div className={`markdown-body text-[15px] ${streaming ? 'stream-cursor' : ''}`}>
        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]} components={{
          table: ({ children }) => <div className="table-scroll"><table>{children}</table></div>,
          img: ({ src, alt }) => <SafeImage src={src} alt={alt} />,
        }}>{text}</ReactMarkdown>
      </div>
      {/* 操作须支持键盘聚焦；生成中禁用重跑。 */}
      <div className="mt-3 flex items-center gap-1 opacity-100 transition focus-within:opacity-100 sm:opacity-0 sm:group-hover:opacity-100">
      <button
        type="button"
        onClick={handleCopy}
        className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-[12px] text-zinc-500 transition hover:bg-zinc-100 hover:text-zinc-800"
        aria-label={copied ? '已复制' : '复制回复'}
      >
        {copied ? <Check size={14} /> : <Copy size={14} />}{copied ? '已复制' : '复制'}
      </button>
      {!streaming && onRegenerate && <button type="button" onClick={onRegenerate} className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-[12px] text-zinc-500 transition hover:bg-zinc-100 hover:text-zinc-800"><ArrowClockwise size={14} />重新生成</button>}
      </div>
    </div>
  )
}
