import { useCallback, useRef, useState } from 'react'

import { streamRegenerate, streamTurn, type ExecutionDraft } from '../api/client'
import type { ApprovalOption, SseEvent, TurnData, WorkItem } from '../types'

export interface LivePendingApproval {
  request_id: string
  tool: string
  input: unknown
  reason: string
  options: ApprovalOption[]
}

/* 异步 GET 只能收尾同代、同轮快照。 */
export interface StreamCompletion {
  generation: number
  turnId: string | null
}

export function useSessionStream(sessionId: string | null) {
  const [turn, setTurn] = useState<TurnData | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pendingApproval, setPendingApproval] = useState<LivePendingApproval | null>(null)
  const [replacedTurnId, setReplacedTurnId] = useState<string | null>(null)

  // ref 避免流回调读取旧状态。
  const turnRef = useRef<TurnData | null>(null)
  // 重跑保留用户消息 ID，以新轮替换旧回复。
  const sentUserRef = useRef<TurnData['user_message']>(null)
  // 新模型步出文时，上一段放回工具之前。
  const textStepRef = useRef<{ id: string; index: number; time: number } | null>(null)
  // 用户 ID 仍为 local 时，用旧轮 ID 定位替换目标。
  const replacementRef = useRef<string | null>(null)
  // 发送时立即更新，拒绝重复发送并使旧 GET 失效。
  const streamingRef = useRef(false)
  const generationRef = useRef(0)
  const getStreamGeneration = useCallback(() => generationRef.current, [])
  const hasActiveStream = useCallback(() => streamingRef.current, [])
  const getCompletion = useCallback((generation = generationRef.current): StreamCompletion => (
    { generation, turnId: turnRef.current?.turn_id ?? null }
  ), [])
  // 镜像 ref 避免归约器读取旧审批。
  const pendingApprovalRef = useRef<{ request_id: string } | null>(null)

  const appendItem = useCallback((item: WorkItem) => {
    if (turnRef.current) {
      turnRef.current.work_items = [...turnRef.current.work_items, item]
      setTurn({ ...turnRef.current })
    }
  }, [])

  const applyEvent = useCallback(
    (event: SseEvent) => {
      const cur = turnRef.current
      switch (event.type) {
        case 'turn_started':
          turnRef.current = {
            turn_id: event.turn_id,
            state: 'running',
            started_at: event.started_at,
            ended_at: null,
            active_ms: null,
            user_message: sentUserRef.current,
            work_items: [],
            final_text: null,
          }
          textStepRef.current = null
          setReplacedTurnId(replacementRef.current)
          setTurn({ ...turnRef.current })
          break

        case 'delta':
          if (cur && event.text) {
            const stepId = event.message_id ?? 'legacy'
            const previous = textStepRef.current
            if (previous?.id !== stepId) {
              // 下一步有正文才移动上一段，否则保留为最终或部分输出。
              if (previous && cur.final_text) {
                const item: WorkItem = { kind: 'text', id: previous.id, text: cur.final_text, time: previous.time }
                cur.work_items = [...cur.work_items.slice(0, previous.index), item, ...cur.work_items.slice(previous.index)]
              }
              cur.final_text = null
              textStepRef.current = { id: stepId, index: cur.work_items.length, time: Date.now() }
            }
            cur.final_text = (cur.final_text ?? '') + event.text
            setTurn({ ...cur })
          }
          break

        case 'tool_started':
          appendItem({
            kind: 'tool_call',
            id: event.tool_call_id,
            name: event.name,
            input: event.input,
            status: 'running',
            output: '',
            time: Date.now(),
          })
          break

        case 'tool_completed':
          if (cur) {
            // 同 ID 更新已有工具卡，避免重复。
            const idx = cur.work_items.findIndex(
              (x) => x.kind === 'tool_call' && x.id === event.tool_call_id,
            )
            const previous = cur.work_items[idx]
            const item: WorkItem = {
              kind: 'tool_call',
              id: event.tool_call_id,
              name: event.name,
              // 完成事件不得覆盖已有参数和顺序时间。
              input: previous?.kind === 'tool_call' ? previous.input : undefined,
              status: event.status,
              output: event.output_preview,
              time: previous?.time ?? Date.now(),
            }
            cur.work_items =
              idx >= 0
                ? cur.work_items.map((x, i) => (i === idx ? item : x))
                : [...cur.work_items, item]
            setTurn({ ...cur })
          }
          break

        case 'todo_updated':
          appendItem({
            kind: 'todo',
            id: `todo-${Date.now()}`,
            items: event.items,
            time: Date.now(),
          })
          break

        case 'subtask_started':
          appendItem({
            kind: 'subtask',
            id: event.subtask_id,
            goal: event.goal,
            status: 'running',
            output: '',
            time: Date.now(),
          })
          break

        case 'subtask_delta':
          if (cur) {
            const idx = [...cur.work_items].reverse().findIndex(
              (x) => x.kind === 'subtask' && x.id === event.subtask_id,
            )
            if (idx >= 0) {
              const real = cur.work_items.length - 1 - idx
              const sub = cur.work_items[real] as Extract<WorkItem, { kind: 'subtask' }>
              sub.output += event.text
              setTurn({ ...cur })
            }
          }
          break

        case 'subtask_completed':
          if (cur) {
            // 原地更新，保持右栏持有的条目引用有效。
            const sub = cur.work_items.find(
              (x) => x.kind === 'subtask' && x.id === event.subtask_id,
            )
            if (sub && sub.kind === 'subtask') sub.status = event.status
            setTurn({ ...cur })
          }
          break

        case 'approval_request':
          appendItem({
            kind: 'approval',
            id: event.request_id,
            tool: event.tool,
            input: event.input,
            reason: event.reason,
            status: 'requested',
            time: Date.now(),
          })
          setPendingApproval({
            request_id: event.request_id,
            tool: event.tool,
            input: event.input,
            reason: event.reason,
            options: event.options,
          })
          pendingApprovalRef.current = { request_id: event.request_id }
          break

        case 'approval_resolved':
          if (cur) {
            cur.work_items = cur.work_items.map((x) =>
              x.kind === 'approval' && x.id === event.request_id
                ? { ...x, status: event.approved ? 'approved' : 'denied' }
                : x,
            )
            setTurn({ ...cur })
          }
          if (pendingApprovalRef.current?.request_id === event.request_id) {
            pendingApprovalRef.current = null
            setPendingApproval(null)
          }
          break

        case 'compacted':
          appendItem({
            kind: 'compaction',
            id: `compact-${Date.now()}`,
            tokens_before: event.tokens_before,
            tokens_after: event.tokens_after,
            time: Date.now(),
          })
          break

        case 'error':
          setError(event.message)
          appendItem({ kind: 'error', id: `error-${Date.now()}`, message: event.message, time: Date.now() })
          break

        case 'turn_completed':
          if (turnRef.current) {
            turnRef.current.state = event.state
            turnRef.current.ended_at = event.ended_at
            turnRef.current.active_ms = event.active_ms
            setTurn({ ...turnRef.current })
          }
          break

        default:
          break
      }
    },
    [appendItem],
  )

  const send = useCallback(
    async (text: string, onFinished?: (completion: StreamCompletion) => void, execution?: ExecutionDraft, onAccepted?: () => void) => {
      if (!sessionId || streamingRef.current) return
      streamingRef.current = true
      const generation = ++generationRef.current
      setError(null)
      setPendingApproval(null)
      pendingApprovalRef.current = null
      setIsStreaming(true)
      sentUserRef.current = { id: 'local', text, time: Date.now() }
      replacementRef.current = null
      // 建轮前保留旧身份，拒绝发送后仍可恢复旧轮。

      try {
        await streamTurn(sessionId, text, (event) => {
          applyEvent(event)
          // 建轮确认前保留草稿，失败时可重发。
          if (event.type === 'turn_started') onAccepted?.()
        }, execution)
      } catch (e) {
        // 断线不取消后端轮次。
        setError(e instanceof Error ? e.message : '连接中断，可稍后刷新回看进度')
      } finally {
        streamingRef.current = false
        setIsStreaming(false)
        onFinished?.(getCompletion(generation))
      }
    },
    [sessionId, applyEvent, getCompletion],
  )

  /* 同轮回放同步审批，仅终态清除快照；活跃流仍由 SSE 管理。 */
  const reconcileTurn = useCallback((completion: StreamCompletion, replay: TurnData, pending: LivePendingApproval | null) => {
    if (streamingRef.current || generationRef.current !== completion.generation
        || turnRef.current?.turn_id !== completion.turnId
        || replay.turn_id !== completion.turnId) return
    // 立即采用服务端下一审批或空值，避免旧弹窗占位。
    setPendingApproval(pending)
    pendingApprovalRef.current = pending
    if (replay.state === 'running') return
    // 同代同轮的终态回放可收尾漏收完成事件的快照。
    turnRef.current = null
    setTurn(null)
    setPendingApproval(null)
    pendingApprovalRef.current = null
    setReplacedTurnId(null)
  }, [])

  const regenerate = useCallback(
    async (source: TurnData | null, onFinished?: (completion: StreamCompletion) => void) => {
      if (!sessionId || streamingRef.current) return
      streamingRef.current = true
      const generation = ++generationRef.current
      setError(null)
      setPendingApproval(null)
      pendingApprovalRef.current = null
      setIsStreaming(true)
      sentUserRef.current = source?.user_message ?? null
      replacementRef.current = source?.turn_id ?? null
      // 重跑被拒绝时保留旧快照及身份，继续恢复终态。

      try {
        await streamRegenerate(sessionId, applyEvent)
      } catch (e) {
        setError(e instanceof Error ? e.message : '连接中断，可稍后刷新回看进度')
      } finally {
        streamingRef.current = false
        setIsStreaming(false)
        onFinished?.(getCompletion(generation))
      }
    },
    [sessionId, applyEvent, getCompletion],
  )

  return { turn, isStreaming, error, send, regenerate, reconcileTurn, pendingApproval, replacedTurnId, getStreamGeneration, hasActiveStream, getCompletion }
}
