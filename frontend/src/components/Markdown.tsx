import { useState } from 'react'
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
}: {
  text: string
  streaming?: boolean
  onRegenerate?: () => void
}) {
  return (
    <div className="group relative min-w-0">
      <div className={`markdown-body text-[15px] ${streaming ? 'stream-cursor' : ''}`}>
        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]} components={{
          table: ({ children }) => <div className="table-scroll"><table>{children}</table></div>,
          img: ({ src, alt }) => <SafeImage src={src} alt={alt} />,
        }}>{text}</ReactMarkdown>
      </div>
    </div>
  )
}
