import { useEffect, useState } from 'react'
import { UsersThree, X } from '@phosphor-icons/react'
import { MarkdownWithCopy } from './MessageItem'

import type { WorkItem } from '../types'

export type SubtaskItem = Extract<WorkItem, { kind: 'subtask' }>

export interface SubtaskViewerProps {
  subtask: SubtaskItem
  onClose: () => void
}

export function SubtaskViewer({ subtask, onClose }: SubtaskViewerProps) {
  const running = subtask.status === 'running'
  // 输出原地追加，定时刷新以读取最新值。
  const [, setTick] = useState(0)
  useEffect(() => {
    if (!running) return
    const timer = setInterval(() => setTick((t) => t + 1), 500)
    return () => clearInterval(timer)
  }, [running])

  return (
    <aside aria-label="子助手详情" className="subtask-panel flex w-[360px] shrink-0 flex-col border-l border-zinc-200 bg-white">
      <div className="flex h-[68px] shrink-0 items-center gap-2.5 border-b border-zinc-200 px-5 text-[14px] font-semibold text-zinc-800">
        <UsersThree size={20} className="text-accent-700" />子助手
        <button type="button" onClick={onClose} className="icon-button ml-auto" aria-label="关闭面板" title="关闭面板"><X size={18} /></button>
      </div>
      <div className="m-5 rounded-xl border border-zinc-200/70 bg-zinc-50 p-4">
        <div className="flex items-start justify-between gap-2">
          <p className="text-[13px] font-medium leading-snug text-zinc-800">{subtask.goal}</p>
        </div>
        <p
          className={`mt-1 text-[11px] ${
            running ? 'text-accent-700' : subtask.status === 'failed' ? 'text-red-600' : 'text-accent-700'
          }`}
        >
          {running ? '执行中' : subtask.status === 'failed' ? '执行失败' : subtask.status === 'stopped' ? '已停止' : '已完成'}
        </p>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 pb-6">
        {subtask.output ? (
          <MarkdownWithCopy text={subtask.output} streaming={running} />
        ) : (
          <p role="status" className="text-[13px] text-zinc-500">等待子助手输出…</p>
        )}
      </div>
    </aside>
  )
}
