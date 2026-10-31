import type { ApprovalOptionId, SessionDetail, SessionItem, SseEvent } from '../types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
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
