import { useCallback, useEffect, useRef, useState } from 'react'
import { ChatTeardropText, GearSix, ShieldCheck, Trash } from '@phosphor-icons/react'

import { ApiError, cancelAssistantReset, decideChatApproval, getChat, resetAssistant, sendChatMessage, uploadChatFile, type ChatApproval, type ChatSnapshot, type ExecutionDraft } from './api/client'
import { Composer } from './components/Composer'
import { MarkdownWithCopy, UserBubble } from './components/MessageItem'
import { TokenBadge } from './components/TokenBadge'
import { createWriteQueue } from './utils/writeQueue'
import type { ApprovalOptionId } from './types'
import './workspace.css'

export default function App() {
  const [chat, setChat] = useState<ChatSnapshot | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [menuOpen, setMenuOpen] = useState(false)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [resetBusy, setResetBusy] = useState(false)
  const [resetError, setResetError] = useState<string | null>(null)
  const [enqueue] = useState(() => createWriteQueue())
  const retryIds = useRef(new Map<string, string>())
  const sending = useRef(new Set<string>())
  const bottom = useRef<HTMLDivElement>(null)
  const menu = useRef<HTMLDivElement>(null)
  const settingsButton = useRef<HTMLButtonElement>(null)
  const resetId = useRef<string | null>(null)
  const previousAssistant = useRef<string | null>(null)

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
    if (!chat) return
    if (previousAssistant.current && previousAssistant.current !== chat.assistant_id) {
      setDialogOpen(false)
      setResetError(null)
      resetId.current = null
    }
    previousAssistant.current = chat.assistant_id
  }, [chat])

  useEffect(() => {
    if (!menuOpen) return
    const onDown = (event: MouseEvent) => { if (!menu.current?.contains(event.target as Node)) setMenuOpen(false) }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [menuOpen])

  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    bottom.current?.scrollIntoView({ behavior: reduced ? 'instant' : 'smooth', block: 'end' })
  }, [chat?.messages, chat?.approvals])

  const send = (text: string, execution: ExecutionDraft, onAccepted?: (remaining?: File[]) => void, files: File[] = []) => {
    if (!chat || chat.reset || resetBusy) return
    const assistantId = chat.assistant_id
    const key = JSON.stringify([assistantId, text, execution.mode, files.map((f) => [f.name, f.size, f.lastModified])])
    if (sending.current.has(key)) return
    sending.current.add(key)
    const id = retryIds.current.get(key) ?? crypto.randomUUID()
    const submit = async () => {
      const uploaded = new Map<string, string>()
      const errors: string[] = []
      let status = 0
      for (const [index, file] of files.entries()) {
        try {
          const result = await uploadChatFile(assistantId, `${id}-${index}`, file)
          uploaded.set(result.path, result.upload_id)
        } catch (e) {
          if (e instanceof ApiError) status = e.status
          errors.push(`${file.name}：${e instanceof Error ? e.message : '上传失败'}`)
        }
      }
      if (errors.length) {
        const message = `文件上传失败，请重试或移除后再发送：${errors.join('；')}`
        throw status ? new ApiError(status, message) : new Error(message)
      }
      await enqueue(() => sendChatMessage(assistantId, id, text, execution.mode, [...uploaded.values()]))
    }
    void submit().then(() => {
      retryIds.current.delete(key)
      setError(null)
      if (previousAssistant.current === assistantId) onAccepted?.()
    }).catch((e: unknown) => {
      if (e instanceof ApiError && e.status === 409) {
        retryIds.current.delete(key)
        void getChat().then(update).catch(() => {})
      } else retryIds.current.set(key, id)
      setError(e instanceof Error ? e.message : '发送失败，请重试')
    }).finally(() => { sending.current.delete(key) })
  }

  const changeReset = async (cancel: boolean, confirmStopped = false) => {
    if (!chat || resetBusy) return
    const assistantId = chat.reset?.assistant_id ?? chat.assistant_id
    const id = chat.reset?.id ?? resetId.current ?? crypto.randomUUID()
    resetId.current = id
    setResetBusy(true)
    setResetError(null)
    try {
      if (cancel) await cancelAssistantReset(assistantId, id)
      else await resetAssistant(assistantId, id, confirmStopped)
      const snapshot = await getChat()
      update(snapshot)
      if (cancel && !snapshot.reset) { setDialogOpen(false); resetId.current = null }
    } catch (e) {
      setResetError(e instanceof Error ? e.message : '操作失败，请重试')
      if (e instanceof ApiError && e.status === 409) {
        try {
          const snapshot = await getChat()
          update(snapshot)
          if (cancel && !snapshot.reset) { setDialogOpen(false); resetId.current = null }
        } catch { /* 保留请求 ID，稍后可重试。 */ }
      }
    } finally {
      setResetBusy(false)
    }
  }

  const closeDialog = () => {
    if (chat?.reset || resetId.current) void changeReset(true)
    else if (!resetBusy) { setDialogOpen(false); setResetError(null); resetId.current = null; settingsButton.current?.focus() }
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
          <div ref={menu} className="relative" onKeyDown={(event) => {
            if (event.key === 'Escape') { setMenuOpen(false); settingsButton.current?.focus() }
          }}>
            <button ref={settingsButton} type="button" className="icon-button" aria-label="设置" aria-expanded={menuOpen}
              aria-controls="assistant-settings" onClick={() => setMenuOpen(!menuOpen)}><GearSix size={20} /></button>
            {menuOpen && <div id="assistant-settings" role="group" aria-label="助手设置" className="floating-menu absolute right-0 top-full z-20 mt-2 w-40 rounded-xl bg-white p-2">
              <button type="button" disabled={!chat || !!chat.reset} onClick={() => { setMenuOpen(false); setDialogOpen(true) }}
                className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-[13px] text-red-700 hover:bg-red-50 disabled:opacity-50"><Trash size={16} />删除</button>
            </div>}
          </div>
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-8 sm:px-8">
          {chat && (chat.messages.length > 0 || chat.approvals.length > 0) ? (
            <div className="mx-auto max-w-[800px] space-y-10">
              {timeline.map((item) => item.kind === 'approval'
                ? <ApprovalCard key={item.approval.request_id} approval={item.approval}
                    disabled={!!chat?.reset || resetBusy}
                    decide={(option, feedback) => enqueue(() => decideChatApproval(chat!.assistant_id, item.approval.request_id, option, feedback))
                      .catch((e: unknown) => { if (e instanceof ApiError && e.status === 409) void getChat().then(update).catch(() => {}); throw e })} />
                : item.message.role === 'user'
                ? <UserBubble key={item.message.id} text={item.message.text} attachments={item.message.attachments} />
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
        <Composer disabled={!chat || !!chat.reset || resetBusy} streaming={false} assistantId={chat?.assistant_id} permission={{ mode: chat?.permission_mode ?? 'build' }}
          onSend={send} onStop={() => {}} />
      </main>
      {(dialogOpen || chat?.reset) && <ResetDialog reset={chat?.reset ?? null} busy={resetBusy} error={resetError}
        onCancel={closeDialog} onConfirm={(confirmStopped) => void changeReset(false, confirmStopped)} />}
    </div>
  )
}

function ResetDialog({ reset, busy, error, onCancel, onConfirm }: {
  reset: ChatSnapshot['reset']
  busy: boolean
  error: string | null
  onCancel: () => void
  onConfirm: (confirmStopped: boolean) => void
}) {
  const dialog = useRef<HTMLDivElement>(null)
  const cancel = useRef(onCancel)
  useEffect(() => { cancel.current = onCancel }, [onCancel])
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    dialog.current?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); cancel.current(); return }
      if (event.key !== 'Tab' || !dialog.current) return
      const controls = Array.from(dialog.current.querySelectorAll<HTMLElement>('button:not(:disabled)'))
      if (!controls.length) { event.preventDefault(); dialog.current.focus(); return }
      const first = controls[0], last = controls[controls.length - 1]
      if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) {
        event.preventDefault(); last.focus()
      } else if (!event.shiftKey && (document.activeElement === last || document.activeElement === dialog.current)) {
        event.preventDefault(); first.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => { window.removeEventListener('keydown', onKey); if (previous?.isConnected) previous.focus() }
  }, [])
  const unknown = reset?.reason === 'unknown'
  return <div className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-900/30 p-4 backdrop-blur-[3px]">
    <div ref={dialog} role="dialog" aria-modal="true" aria-labelledby="reset-title" aria-describedby="reset-description"
      tabIndex={-1} className="approval-dialog max-h-[90dvh] w-full max-w-[520px] overflow-y-auto rounded-[20px] border border-zinc-200 bg-white p-6 shadow-[0_24px_80px_#30261826]">
      <h2 id="reset-title" className="text-[18px] font-semibold text-zinc-900">删除助手</h2>
      <div id="reset-description" className="mt-4 space-y-3 text-[13px] leading-6 text-zinc-600">
        <p>将停止正在运行的任务，归档当前对话、任务和结果，并自动创建全新的助手。此版本暂不提供归档查看入口。</p>
        <p>已开始的操作可能产生的影响无法撤销。工作区文件、项目权限规则、模型连接配置和模型调用日志会保留；日志可能包含旧对话。</p>
        <p>权限模式会恢复为“变更前确认”，本会话授权会清除。新助手不会自动继承旧对话和任务，仍可按权限访问共享工作区。</p>
      </div>
      {reset && <p role="status" className="mt-4 rounded-lg bg-amber-50 px-3 py-2 text-[13px] leading-6 text-amber-800">
        {reset.status === 'stopping' ? '正在等待任务和已开始的操作停止…'
          : unknown ? '重启前有操作的结果未知。请检查并确认该操作已停止，再继续删除。'
          : reset.reason === 'timeout' ? '仍有操作未停止，删除尚未完成。可稍后重试或取消。'
          : '删除未完成，请重试或取消。'}
        <span className="mt-1 block">取消删除后，已停止的任务不会自动恢复。</span>
      </p>}
      {error && <p role="alert" className="mt-3 text-[13px] text-red-700">{error}</p>}
      <div className="mt-6 flex flex-wrap justify-end gap-3">
        <button type="button" disabled={busy} onClick={onCancel} className="rounded-xl border border-zinc-300 px-4 py-2.5 text-[13px] text-zinc-700 hover:bg-zinc-50 disabled:opacity-50">取消</button>
        {(!reset || reset.status === 'blocked') && <button type="button" disabled={busy} onClick={() => onConfirm(unknown)}
          className="rounded-xl bg-red-700 px-4 py-2.5 text-[13px] font-semibold text-white hover:bg-red-800 disabled:opacity-50">
          {busy ? '处理中…' : unknown ? '确认操作已停止并继续' : reset ? '重试删除' : '删除并新建'}
        </button>}
      </div>
    </div>
  </div>
}

function ApprovalCard({ approval, decide, disabled }: {
  approval: ChatApproval
  disabled: boolean
  decide: (option: ApprovalOptionId, feedback?: string) => Promise<unknown>
}) {
  const [busy, setBusy] = useState(false)
  const [feedback, setFeedback] = useState('')
  const [error, setError] = useState<string | null>(null)
  const busyRef = useRef(false)
  const options = approval.options.filter((option) => ['allowOnce', 'allowSession', 'deny'].includes(option.option_id))
  const select = (option: ApprovalOptionId) => {
    if (busyRef.current || disabled) return
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
        {options.map((option) => <button key={option.option_id} type="button" disabled={busy || disabled}
          onClick={() => select(option.option_id)} title={option.content} className="rounded-lg border border-zinc-300 bg-white px-3 py-2 text-left text-[13px] text-zinc-700 hover:bg-zinc-50 disabled:opacity-50">
          {option.label}
          {option.option_id === 'allowSession' && <span className="mt-1 block text-[11px] text-zinc-500">仅限当前会话，后端重启后失效</span>}
          {option.content && <span className="mt-1 block break-all font-mono text-[11px] text-zinc-500">{option.content}</span>}
        </button>)}
      </div>
      <label className="mt-3 block text-[12px] text-zinc-600">拒绝理由（可选）
        <textarea value={feedback} onChange={(e) => setFeedback(e.target.value)} maxLength={4096} rows={2} disabled={disabled}
          className="mt-1 block w-full resize-y rounded-lg border border-zinc-300 bg-white p-2 text-[13px]" />
      </label>
      {error && <p role="alert" className="mt-2 text-[13px] text-red-700">{error}</p>}
    </section>
  )
}
