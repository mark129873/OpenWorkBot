import { Suspense, lazy } from 'react'
import { FileText } from '@phosphor-icons/react'

const LazyMarkdown = lazy(() => import('./Markdown'))

export function MarkdownWithCopy({ text, streaming = false, onRegenerate }: { text: string; streaming?: boolean; onRegenerate?: () => void }) {
  return (
    /* 等高占位避免首帧跳动。 */
    <Suspense fallback={<div role="status" className="text-[13px] leading-relaxed text-zinc-500">正在加载正文…</div>}>
      <LazyMarkdown text={text} streaming={streaming} onRegenerate={onRegenerate} />
    </Suspense>
  )
}

export function UserBubble({ text, attachments = [] }: { text: string; attachments?: { id: string; name: string; path: string }[] }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[88%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md border border-accent-100/70 bg-accent-50 px-5 py-3.5 text-[15px] leading-7 text-zinc-800">
        {text}
        {attachments.map((item) => <div key={item.id} className="flex items-center gap-2 text-[13px] text-zinc-700">
          <FileText size={16} aria-hidden="true" /><span>{item.name}</span>
        </div>)}
      </div>
    </div>
  )
}
