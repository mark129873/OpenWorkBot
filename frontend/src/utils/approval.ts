import type { SessionDetail } from '../types'
import type { PendingApproval } from '../components/ApprovalModal'

/* 审批以服务端未决请求为准，不信任前端缓存。 */
export function pendingFromDetail(detail: SessionDetail | null): PendingApproval | null {
  const p = detail?.pending_approval
  if (!p || p.status !== 'requested') return null
  return { request_id: p.request_id, tool: p.tool, input: p.input, reason: p.reason, options: p.options ?? [] }
}
