import type { ApprovalOption, ApprovalOptionId, SessionDetail, SessionItem, SseEvent } from '../types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

let csrfToken = ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  if (!(init?.body instanceof FormData) && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  if (init?.method && !['GET', 'HEAD'].includes(init.method.toUpperCase())) headers.set('X-CSRF-Token', csrfToken)
  const resp = await fetch(path, {
    ...init,
    headers,
    credentials: 'same-origin',
  })
  if (!resp.ok) {
    let detail = `请求失败（${resp.status}）`
    try {
      const body = await resp.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      /* 解析失败保留默认错误信息。 */
    }
    throw new ApiError(resp.status, detail)
  }
  return resp.json() as Promise<T>
}

export interface AuthSession { username: string | null; csrf: string }

export async function getAuthSession(): Promise<AuthSession> {
  const result = await request<AuthSession>('/api/auth/session')
  csrfToken = result.csrf
  return result
}

export async function authenticate(username: string, password: string, register: boolean): Promise<AuthSession> {
  const result = await request<AuthSession>(`/api/auth/${register ? 'register' : 'login'}`, {
    method: 'POST', body: JSON.stringify({ username, password }),
  })
  csrfToken = result.csrf
  return result
}

export async function logout(): Promise<void> {
  await request('/api/auth/logout', { method: 'POST' })
  csrfToken = ''
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  status: string
  time: number
  order: number
  attachments?: { id: string; name: string; path: string; size: number }[]
}

export interface ChatApproval {
  request_id: string
  task_id: string | null
  task_title: string | null
  kind?: 'sandbox_network'
  tool: string
  input: unknown
  reason: string
  options: ApprovalOption[]
  order: number
}

export interface ChatSnapshot {
  revision: number
  assistant_id: string
  reset: { id: string; assistant_id: string; status: 'stopping' | 'deleting' | 'blocked'; deletion_started?: boolean; reason?: 'unknown' | 'timeout' | 'error' | null } | null
  messages: ChatMessage[]
  next_before: number | null
  approvals: ChatApproval[]
  permission_mode: string
  context_usage: { used: number; window: number }
}

export function getChat(): Promise<ChatSnapshot> {
  return request('/api/chat?limit=50')
}

export function getChatPage(before: number): Promise<ChatSnapshot> {
  return request(`/api/chat?limit=50&before=${before}`)
}

export function sendChatMessage(assistantId: string, clientMessageId: string, text: string, mode?: string, attachments: string[] = []): Promise<unknown> {
  return request('/api/chat/messages', { method: 'POST', body: JSON.stringify({
    assistant_id: assistantId, client_message_id: clientMessageId, text, attachments, ...(mode ? { permission_mode: mode } : {}),
  }) })
}

export function uploadChatFile(assistantId: string, uploadId: string, file: File): Promise<{ upload_id: string; name: string; path: string }> {
  const body = new FormData()
  body.append('assistant_id', assistantId)
  body.append('upload_id', uploadId)
  body.append('file', file)
  return request('/api/chat/uploads', { method: 'POST', headers: {}, body })
}

export function decideChatApproval(assistantId: string, requestId: string, optionId: ApprovalOptionId, feedback?: string): Promise<unknown> {
  return request(`/api/chat/approvals/${encodeURIComponent(requestId)}`, { method: 'POST', body: JSON.stringify({
    assistant_id: assistantId, option_id: optionId, ...(feedback ? { feedback } : {}),
  }) })
}

export function resetAssistant(assistantId: string, resetId: string, confirmStopped = false): Promise<unknown> {
  return request('/api/chat/reset', { method: 'POST', body: JSON.stringify({
    assistant_id: assistantId, reset_id: resetId, confirm_stopped: confirmStopped,
  }) })
}

export function cancelAssistantReset(assistantId: string, resetId: string): Promise<unknown> {
  return request('/api/chat/reset/cancel', { method: 'POST', body: JSON.stringify({
    assistant_id: assistantId, reset_id: resetId,
  }) })
}

/* 仅分配 ID，首条消息才落库。 */
export function createSession(): Promise<{ id: string }> {
  return request('/api/sessions', { method: 'POST' })
}

export function listSessions(): Promise<SessionItem[]> {
  return request('/api/sessions')
}

export function getSessionDetail(id: string): Promise<SessionDetail> {
  return request(`/api/sessions/${id}`)
}

export function deleteSession(id: string): Promise<{ ok: boolean }> {
  return request(`/api/sessions/${id}`, { method: 'DELETE' })
}

export function stopTurn(id: string): Promise<{ ok: boolean }> {
  return request(`/api/sessions/${id}/stop`, { method: 'POST' })
}

export function submitApproval(
  id: string,
  requestId: string,
  optionId: ApprovalOptionId,
  feedback?: string,
): Promise<{ ok: boolean }> {
  return request(`/api/sessions/${id}/approval`, {
    method: 'POST',
    body: JSON.stringify({
      request_id: requestId,
      option_id: optionId,
      ...(feedback ? { feedback } : {}),
    }),
  })
}

export interface PermissionState {
  mode: string
}

export function getPermissionState(): Promise<PermissionState> {
  return request('/api/permission/state')
}

/* 模式随下一次发送生效。 */
export interface ExecutionDraft {
  mode?: string
}

/* 断线不取消后端轮次，可通过回放恢复。 */
export async function streamTurn(
  sessionId: string,
  text: string,
  onEvent: (event: SseEvent) => void,
  execution?: ExecutionDraft,
): Promise<void> {
  const resp = await fetch(`/api/sessions/${sessionId}/turn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, ...(execution ?? {}) }),
  })
  await consumeSse(resp, onEvent)
}

export async function streamRegenerate(
  sessionId: string,
  onEvent: (event: SseEvent) => void,
): Promise<void> {
  const resp = await fetch(`/api/sessions/${sessionId}/regenerate`, { method: 'POST' })
  await consumeSse(resp, onEvent)
}

async function consumeSse(resp: Response, onEvent: (event: SseEvent) => void): Promise<void> {
  if (!resp.ok || !resp.body) {
    let detail = `请求失败（${resp.status}）`
    try {
      const body = await resp.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      /* 解析失败保留默认错误信息。 */
    }
    throw new ApiError(resp.status, detail)
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // 尾段可能不完整，留待下一块数据。
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''
    for (const frame of frames) {
      const event = parseFrame(frame)
      if (event) onEvent(event)
    }
  }
}

function parseFrame(frame: string): SseEvent | null {
  const lines = frame.split('\n')
  let dataLine: string | null = null
  for (const line of lines) {
    if (line.startsWith(':')) continue // 心跳
    if (line.startsWith('data: ')) dataLine = line.slice(6)
  }
  if (!dataLine) return null
  try {
    return JSON.parse(dataLine) as SseEvent
  } catch {
    return null
  }
}
