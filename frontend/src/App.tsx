import { useCallback, useEffect, useRef, useState } from 'react'
import { ChatTeardropText, ShieldCheck } from '@phosphor-icons/react'

import { decideChatApproval, getChat, sendChatMessage, type ChatApproval, type ChatSnapshot, type ExecutionDraft } from './api/client'
import { Composer } from './components/Composer'
import { MarkdownWithCopy, UserBubble } from './components/MessageItem'
import { TokenBadge } from './components/TokenBadge'
import { createWriteQueue } from './utils/writeQueue'
import type { ApprovalOptionId } from './types'
import './workspace.css'

export default function App() {
  const [chat, setChat] = useState<ChatSnapshot | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [enqueue] = useState(() => createWriteQueue())
  const retryIds = useRef(new Map<string, string>())
  const sending = useRef(new Set<string>())
  const bottom = useRef<HTMLDivElement>(null)

  const update = useCallback((next: ChatSnapshot) => {
    setChat((current) => !current || next.revision > current.revision ? next : current)
  }, [])

  useEffect(() => {
    let active = true
    void getChat().then((snapshot) => { if (active) update(snapshot) })
      .catch((e: unknown) => { if (active) setError(e instanceof Error ? e.message : '聊天加载失败') })
    const source = new EventSource('/api/chat/events')
    source.onmessage = (event) => {
      try {
        const envelope = JSON.parse(event.data) as { data: ChatSnapshot }
        if (active && envelope.data && typeof envelope.data.revision === 'number') update(envelope.data)
      } catch {
        if (active) setError('更新数据无效，正在重新连接')
      }
    }
    source.onopen = () => { if (active) setError(null) }
    source.onerror = () => { if (active) setError('连接中断，正在重新连接；后台任务状态待恢复') }
    return () => { active = false; source.close() }
  }, [update])

  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    bottom.current?.scrollIntoView({ behavior: reduced ? 'instant' : 'smooth', block: 'end' })
  }, [chat?.messages, chat?.approvals])

  const send = (text: string, execution: ExecutionDraft, onAccepted?: () => void) => {
    const key = JSON.stringify([text, execution.mode])
    if (sending.current.has(key)) return
    sending.current.add(key)
    const id = retryIds.current.get(key) ?? crypto.randomUUID()
    void enqueue(() => sendChatMessage(id, text, execution.mode)).then(() => {
      retryIds.current.delete(key)
      setError(null)
      onAccepted?.()
    }).catch((e: unknown) => {
      retryIds.current.set(key, id)
      setError(e instanceof Error ? e.message : '发送失败，请重试')
    }).finally(() => { sending.current.delete(key) })
  }

  const timeline = [
    ...(chat?.messages ?? []).map((message) => ({ kind: 'message' as const, order: message.order, message })),
    ...(chat?.approvals ?? []).map((approval) => ({ kind: 'approval' as const, order: approval.order, approval })),
  ].sort((a, b) => a.order - b.order)

  return (
    <div className="app-shell flex">
      <main className="flex min-w-0 flex-1 flex-col bg-white">
        <header className="workspace-header">
          <h1 className="min-w-0 flex-1 truncate text-[14px] font-semibold text-zinc-800">助手</h1>
          <TokenBadge used={chat?.context_usage.used} window={chat?.context_usage.window} />
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-8 sm:px-8">
          {chat && (chat.messages.length > 0 || chat.approvals.length > 0) ? (
            <div className="mx-auto max-w-[800px] space-y-10">
              {timeline.map((item) => item.kind === 'approval'
                ? <ApprovalCard key={item.approval.request_id} approval={item.approval}
                    decide={(option, feedback) => enqueue(() => decideChatApproval(item.approval.request_id, option, feedback))} />
                : item.message.role === 'user'
                ? <UserBubble key={item.message.id} text={item.message.text} />
                : <div key={item.message.id} className="max-w-[88%]">
                    <p className="mb-2 text-[12px] font-medium text-zinc-500">助手</p>
                    <MarkdownWithCopy text={item.message.text} streaming={item.message.status === 'streaming'} />
                    {item.message.status === 'interrupted' && <p className="mt-2 text-xs text-zinc-500">已中断</p>}
                  </div>)}
            </div>
          ) : (
            <div className="flex h-full flex-col items-center justify-center px-4 text-center">
              <span className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-accent-50 text-accent-700"><ChatTeardropText size={26} weight="duotone" /></span>
              <h2 className="text-[22px] font-semibold tracking-tight text-zinc-900">这次，想完成什么？</h2>
              <p className="mt-3 text-[14px] leading-7 text-zinc-600">描述目标，或从一个具体问题开始。</p>
            </div>
          )}
          <div ref={bottom} />
        </div>
        {error && <div role="alert" className="mx-5 mb-2 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-[13px] text-amber-800">{error}</div>}
        <Composer disabled={!chat} streaming={false} permission={{ mode: chat?.permission_mode ?? 'build' }}
          onSend={send} onStop={() => {}} />
      </main>
    </div>
  )
}

function ApprovalCard({ approval, decide }: {
  approval: ChatApproval
  decide: (option: ApprovalOptionId, feedback?: string) => Promise<unknown>
}) {
  const [busy, setBusy] = useState(false)
  const [feedback, setFeedback] = useState('')
  const [error, setError] = useState<string | null>(null)
  const busyRef = useRef(false)
  const options = approval.options.filter((option) => ['allowOnce', 'allowSession', 'deny'].includes(option.option_id))
  const select = (option: ApprovalOptionId) => {
    if (busyRef.current) return
    busyRef.current = true
    setBusy(true)
    setError(null)
    void decide(option, option === 'deny' ? feedback.trim() || undefined : undefined).catch((e: unknown) => {
      setError(e instanceof Error ? e.message : '提交失败，请重试')
      busyRef.current = false
      setBusy(false)
    })
  }
  return (
    <section aria-label={`任务 ${approval.task_title} 的审批`} className="rounded-xl border border-amber-200 bg-amber-50/40 p-5">
      <h2 className="flex items-center gap-2 text-[14px] font-semibold text-zinc-800"><ShieldCheck size={20} />任务「{approval.task_title}」需要确认</h2>
      <p className="mt-2 text-[13px] leading-6 text-zinc-600">{approval.reason}（{approval.tool}）</p>
      <pre className="mt-3 max-h-40 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-white p-3 text-[12px] text-zinc-700">{JSON.stringify(approval.input ?? {}, null, 2)}</pre>
      <div className="mt-3 flex flex-wrap gap-2">
        {options.map((option) => <button key={option.option_id} type="button" disabled={busy}
          onClick={() => select(option.option_id)} title={option.content} className="rounded-lg border border-zinc-300 bg-white px-3 py-2 text-left text-[13px] text-zinc-700 hover:bg-zinc-50 disabled:opacity-50">
          {option.label}
          {option.option_id === 'allowSession' && <span className="mt-1 block text-[11px] text-zinc-500">仅限当前会话，后端重启后失效</span>}
          {option.content && <span className="mt-1 block break-all font-mono text-[11px] text-zinc-500">{option.content}</span>}
        </button>)}
      </div>
      <label className="mt-3 block text-[12px] text-zinc-600">拒绝理由（可选）
        <textarea value={feedback} onChange={(e) => setFeedback(e.target.value)} maxLength={4096} rows={2}
          className="mt-1 block w-full resize-y rounded-lg border border-zinc-300 bg-white p-2 text-[13px]" />
      </label>
      {error && <p role="alert" className="mt-2 text-[13px] text-red-700">{error}</p>}
    </section>
  )
}
