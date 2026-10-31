import { useEffect, useRef } from 'react'
import { ChatTeardropText, SidebarSimple } from '@phosphor-icons/react'

import { ApprovalModal } from './ApprovalModal'
import { pendingFromDetail } from '../utils/approval'
import { TokenBadge } from './TokenBadge'
import { TurnGroup } from './TurnGroup'
import { Composer } from './Composer'
import { useShell } from '../ShellContext'
import type { LivePendingApproval } from '../hooks/useSessionStream'
import type { ExecutionDraft, PermissionState } from '../api/client'
import type { PendingApproval } from './ApprovalModal'
import type { SubtaskItem } from './SubtaskViewer'
import type { SessionDetail, TurnData } from '../types'

export interface ChatAreaProps {
  detail: SessionDetail | null
  liveTurn: TurnData | null
  /* 原轮次用于本地消息尚无持久 ID 时去重。 */
  replacedTurnId?: string | null
  isStreaming: boolean
  currentId: string | null
  streamError: string | null
  permission: PermissionState
  livePendingApproval: LivePendingApproval | null
  onSend: (text: string, execution: ExecutionDraft, onAccepted?: () => void) => void
  onStop: () => void
  onRegenerate: () => void
  onApprovalResolved: () => void
  onOpenSubtask?: (s: SubtaskItem) => void
}

export function ChatArea(props: ChatAreaProps) {
  const {
    detail, liveTurn, replacedTurnId, isStreaming, currentId, streamError, livePendingApproval, permission,
    onSend, onStop, onRegenerate, onApprovalResolved, onOpenSubtask,
  } = props
  const { sidebarOpen, toggleSidebar } = useShell()
  const bottomRef = useRef<HTMLDivElement>(null)

  // 实时轮替换同轮回放或重跑原轮，避免重复消息。
  const turns = (detail?.turns ?? []).filter((turn) => !liveTurn || (
    turn.turn_id !== liveTurn.turn_id
    && turn.turn_id !== replacedTurnId
    && !(liveTurn.user_message?.id !== 'local' && liveTurn.user_message?.id
      && turn.user_message?.id === liveTurn.user_message.id)
  ))
  const hasContent = turns.length > 0 || liveTurn != null

  // 实时审批优先，否则恢复回放中的未决审批。
  const modalApproval: PendingApproval | null =
    livePendingApproval ?? pendingFromDetail(detail)

  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    bottomRef.current?.scrollIntoView({ behavior: reduced ? 'instant' : 'smooth', block: 'end' })
  }, [turns.length, liveTurn, liveTurn?.final_text, liveTurn?.work_items.length])

  return (
    <main className="flex min-w-0 flex-1 flex-col bg-white">
      <header className="workspace-header">
        <button
          type="button"
          onClick={toggleSidebar}
          className="icon-button"
          aria-label={sidebarOpen ? '收起侧栏' : '展开侧栏'}
          aria-pressed={sidebarOpen}
          title={sidebarOpen ? '收起侧栏' : '展开侧栏'}
        >
          <SidebarSimple size={20} weight="regular" />
        </button>
        <h1 className="min-w-0 flex-1 truncate text-[14px] font-semibold text-zinc-800">
          {detail?.session.title || '新会话'}
          {liveTurn?.state === 'running' && (
            <span className="ml-2 text-xs font-normal text-accent-700">生成中…</span>
          )}
        </h1>
        <TokenBadge used={detail?.session.context_used} window={detail?.session.context_window} />
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-8 sm:px-8">
        {hasContent ? (
          <div className="mx-auto max-w-[800px] space-y-10">
            {turns.map((t) => (
              <TurnGroup
                key={t.turn_id}
                turn={t}
                onOpenSubtask={onOpenSubtask}
                onRegenerate={onRegenerate}
              />
            ))}
            {liveTurn && (
              <TurnGroup turn={liveTurn} onOpenSubtask={onOpenSubtask} onRegenerate={onRegenerate} />
            )}
          </div>
        ) : (
          <div className="flex h-full flex-col items-center justify-center px-4 text-center">
            <span className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-accent-50 text-accent-700"><ChatTeardropText size={26} weight="duotone" /></span>
            <h2 className="text-[22px] font-semibold tracking-tight text-zinc-900">这次，想完成什么？</h2>
            <p className="mt-3 text-[14px] leading-7 text-zinc-600">发送第一条消息，开始这段对话。<br />可以描述目标，也可以从一个具体问题开始。</p>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {streamError && (
        <div role="alert" className="mx-5 mb-2 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-[13px] text-amber-800">
          {streamError}
        </div>
      )}

      <Composer
        disabled={!currentId}
        streaming={isStreaming}
        permission={permission}
        onSend={onSend}
        onStop={onStop}
      />

      {modalApproval && currentId && (
        <ApprovalModal
          sessionId={currentId}
          approval={modalApproval}
          onResolved={onApprovalResolved}
        />
      )}
    </main>
  )
}
