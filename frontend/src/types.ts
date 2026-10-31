/* 会话授权由后端保存，不改变项目模式。 */
export type ApprovalOptionId = 'allowOnce' | 'allowSession' | 'deny'
export interface ApprovalOption { option_id: ApprovalOptionId; label: string; content?: string }

/* 时间字段沿用后端 epoch 毫秒。 */

export interface SessionItem {
  id: string
  title: string
  model: string
  created_at: number
  updated_at: number
  tokens_used: number
  input_tokens: number
  running: boolean
}

export type WorkItem =
  | { kind: 'text'; id: string; text: string; time: number }
  | {
      kind: 'tool_call'
      id: string
      name: string
      input: unknown
      status: ToolStatus
      output: string
      time: number
    }
  | {
      kind: 'subtask'
      id: string
      goal: string
      status: string
      output: string
      time: number
    }
  | { kind: 'todo'; id: string; items: TodoItem[]; time: number }
  | { kind: 'error'; id: string; message: string; time: number }
  | {
      kind: 'approval'
      id: string
      tool: string
      input: unknown
      reason: string
      status: 'requested' | 'approved' | 'denied' | 'auto' // auto 表示权限规则放行
      time: number
    }
  | {
      kind: 'compaction'
      id: string
      tokens_before: number | null
      tokens_after: number | null
      time: number
    }

export type ToolStatus = 'pending' | 'running' | 'completed' | 'failed' | 'denied' | 'stopped'

export interface TodoItem {
  content: string
  status: 'pending' | 'in_progress' | 'completed'
}

export interface TurnData {
  turn_id: string
  state: 'running' | 'success' | 'stopped' | 'failed'
  started_at: number
  ended_at: number | null
  active_ms: number | null
  user_message: { id: string; text: string; time: number } | null
  work_items: WorkItem[]
  final_text: string | null
}

export interface SessionDetail {
  session: {
    id: string
    title: string
    model: string
    created_at: number
    updated_at: number
    tokens_used: number
    input_tokens: number
    context_used: number // 最近一轮最后一步的输入量，近似上下文占用
    context_window: number
  }
  turns: TurnData[]
  pending_approval: {
    turn_id?: string
    request_id: string
    tool: string
    input: unknown
    reason: string
    status: string
    options?: ApprovalOption[]
    time?: number
  } | null
}

export type SseEvent =
  | { type: 'turn_started'; turn_id: string; started_at: number }
  // 缺少 message_id 时无法区分模型步骤。
  | { type: 'delta'; text: string; message_id?: string }
  | { type: 'tool_started'; tool_call_id: string; name: string; input: unknown }
  | {
      type: 'tool_completed'
      tool_call_id: string
      name: string
      status: ToolStatus
      output_preview: string
    }
  | { type: 'todo_updated'; items: TodoItem[] }
  | { type: 'subtask_started'; subtask_id: string; goal: string }
  | { type: 'subtask_delta'; subtask_id: string; text: string }
  | { type: 'subtask_completed'; subtask_id: string; status: string }
  | {
      type: 'approval_request'
      request_id: string
      tool: string
      input: unknown
      reason: string
      options: ApprovalOption[]
    }
  | { type: 'approval_resolved'; request_id: string; approved: boolean }
  | { type: 'token_count'; input_tokens: number; output_tokens: number }
  | { type: 'compacted'; tokens_before: number; tokens_after: number }
  | {
      type: 'turn_completed'
      turn_id: string
      ended_at: number
      active_ms: number
      state: 'success' | 'stopped' | 'failed'
    }
  | { type: 'error'; message: string }
