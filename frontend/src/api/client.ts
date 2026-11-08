import type { ApprovalOption, ApprovalOptionId } from '../types'

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

export interface PermissionState { mode: string }
export interface ExecutionDraft { mode?: string }
