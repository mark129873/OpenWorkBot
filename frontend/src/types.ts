/* 会话授权由后端保存，不改变项目模式。 */
export type ApprovalOptionId = 'allowOnce' | 'allowSession' | 'deny'
export interface ApprovalOption { option_id: ApprovalOptionId; label: string; content?: string }
