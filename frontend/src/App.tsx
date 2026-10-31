import { useCallback, useEffect, useRef, useState } from 'react'
import { Route, Routes, matchPath, useLocation, useNavigate, useParams } from 'react-router-dom'

import { ChatArea } from './components/ChatArea'
import { Sidebar } from './components/Sidebar'
import { SubtaskViewer, type SubtaskItem } from './components/SubtaskViewer'
import { ApiError, createSession, getPermissionState, getSessionDetail, stopTurn, type ExecutionDraft, type PermissionState } from './api/client'
import { ShellContext, useShell } from './ShellContext'
import { useSessionStream, type StreamCompletion } from './hooks/useSessionStream'
import { pendingFromDetail } from './utils/approval'
import { ArrowRight, FileText, ListChecks, SidebarSimple, SquaresFour, TerminalWindow, UsersThree } from '@phosphor-icons/react'
import './workspace.css'
import type { SessionDetail, SessionItem } from './types'

const SIDEBAR_STORAGE_KEY = 'ui.sidebar'

export default function App() {
  return <AppShell />
}

function AppShell() {
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const currentId = matchPath('/session/:sessionId', pathname)?.params.sessionId ?? null
  const [sessions, setSessions] = useState<SessionItem[]>([])
  // 仅本次创建的草稿可在详情 404 时保留空对话。
  const draftIdsRef = useRef<Set<string>>(new Set())

  const [sidebarOpen, setSidebarOpen] = useState(() => {
    try {
      const saved = localStorage.getItem(SIDEBAR_STORAGE_KEY)
      return saved ? saved !== 'collapsed' : !window.matchMedia('(max-width: 760px)').matches
    } catch {
      return true
    }
  })
  const toggleSidebar = useCallback(() => {
    setSidebarOpen((open) => {
      const next = !open
      try {
        localStorage.setItem(SIDEBAR_STORAGE_KEY, next ? 'open' : 'collapsed')
      } catch {
        /* 持久化失败仅影响下次打开。 */
      }
      return next
    })
  }, [])

  const openSession = useCallback(
    (id: string) => navigate(`/session/${id}`),
    [navigate],
  )

  /* 先登记草稿再导航，避免 404 被误判为未知会话。 */
  const handleDraftCreated = useCallback(
    (id: string) => {
      draftIdsRef.current.add(id)
      openSession(id)
    },
    [openSession],
  )

  const handleNewSession = useCallback(async () => {
    const { id } = await createSession()
    handleDraftCreated(id)
  }, [handleDraftCreated])

  return (
    /* 保留 flex 容器，避免侧栏与正文纵向堆叠。 */
    <ShellContext.Provider value={{ sidebarOpen, toggleSidebar }}>
      <div className="app-shell flex">
      <Sidebar
        sessions={sessions}
        currentId={currentId}
        onSelect={openSession}
        onSessionsChange={setSessions}
        onDraftCreated={handleDraftCreated}
      />

        <Routes>
          <Route path="/" element={<WelcomePage onNewSession={handleNewSession} />} />
          <Route
            path="/session/:sessionId"
            element={<SessionRoute draftIdsRef={draftIdsRef} />}
          />
          <Route path="*" element={<WelcomePage onNewSession={handleNewSession} />} />
        </Routes>
      </div>
    </ShellContext.Provider>
  )
}

function WelcomePage({ onNewSession }: { onNewSession: () => Promise<void> }) {
  const { sidebarOpen, toggleSidebar } = useShell()
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const start = async () => {
    if (creating) return
    setCreating(true)
    setError(null)
    try { await onNewSession() }
    catch (e) { setError(e instanceof Error ? e.message : '新建会话失败，请重试') }
    finally { setCreating(false) }
  }
  const capabilities = [
    { icon: FileText, title: '文件工作', text: '阅读、整理与修改工作区文件' },
    { icon: TerminalWindow, title: '命令执行', text: '按你的授权，在本地运行命令' },
    { icon: ListChecks, title: '任务计划', text: '拆分目标，记录每一步工作进展' },
    { icon: UsersThree, title: '子助手协作', text: '将独立工作交给专注的子助手' },
  ]
  return (
    <main className="flex min-w-0 flex-1 flex-col bg-white">
      <header className="workspace-header">
        <SidebarToggle sidebarOpen={sidebarOpen} onToggle={toggleSidebar} />
        <span className="text-[13px] font-medium text-zinc-600">工作台</span>
      </header>
      <div className="flex flex-1 overflow-y-auto">
        <div className="welcome-content animate-enter">
          <div className="mb-7 flex h-14 w-14 items-center justify-center rounded-2xl bg-accent-50 text-accent-700">
            <SquaresFour size={28} weight="duotone" />
          </div>
          <h1 className="welcome-title font-semibold text-zinc-900">从一个目标开始。</h1>
          <p className="mt-4 max-w-md text-[15px] leading-7 text-zinc-600">
            告诉助手你想完成什么。探索文件、安排任务，<br className="hidden sm:block" />
            一起把下一步工作推进下去。
          </p>
          <button type="button" onClick={() => void start()} disabled={creating} className="primary-button mt-7">
            {creating ? '正在新建…' : '新建会话'} <ArrowRight size={17} weight="bold" />
          </button>
          {error && <p role="alert" className="mt-3 text-sm text-red-700">{error}</p>}
          <div className="capability-grid mt-14 border-t border-zinc-200/80 pt-8">
            {capabilities.map(({ icon: Icon, title, text }) => (
              <div key={title} className="flex items-start gap-3.5">
                <Icon size={22} weight="duotone" className="mt-0.5 shrink-0 text-accent-700" />
                <div>
                  <h2 className="text-[14px] font-semibold text-zinc-800">{title}</h2>
                  <p className="mt-1.5 text-[13px] leading-6 text-zinc-600">{text}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      <p className="shrink-0 px-6 pb-6 text-center text-[12px] text-zinc-500">会话保存在本机 · 对话按配置发送至模型 API</p>
    </main>
  )
}

function SidebarToggle({ sidebarOpen, onToggle }: { sidebarOpen: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className="icon-button"
      aria-label={sidebarOpen ? '收起侧栏' : '展开侧栏'}
      aria-pressed={sidebarOpen}
      title={sidebarOpen ? '收起侧栏' : '展开侧栏'}
    >
      <SidebarSimple size={20} weight="regular" />
    </button>
  )
}

/* 按 sessionId 重挂载，隔离会话局部状态。 */
function SessionRoute({
  draftIdsRef,
}: {
  draftIdsRef: React.RefObject<Set<string>>
}) {
  const { sessionId } = useParams()
  if (!sessionId) return null
  return <SessionView key={sessionId} sessionId={sessionId} draftIdsRef={draftIdsRef} />
}

function SessionView({
  sessionId,
  draftIdsRef,
}: {
  sessionId: string
  draftIdsRef: React.RefObject<Set<string>>
}) {
  const navigate = useNavigate()
  const [detail, setDetail] = useState<SessionDetail | null>(null)
  const [activeSubtask, setActiveSubtask] = useState<SubtaskItem | null>(null)
  const [permission, setPermission] = useState<PermissionState>({ mode: 'build' })

  const refreshPermission = useCallback(() => {
    getPermissionState().then(setPermission).catch(() => {})
  }, [])

  useEffect(() => {
    refreshPermission()
  }, [refreshPermission])

  const {
    turn: liveTurn,
    isStreaming,
    error: streamError,
    send,
    regenerate,
    reconcileTurn,
    pendingApproval,
    replacedTurnId,
    getStreamGeneration,
    hasActiveStream,
    getCompletion,
  } = useSessionStream(sessionId)
  // 请求号隔离并发 GET；代次隔离新旧流。
  const detailRequestRef = useRef(0)

  // 仅本次创建的草稿可保留 404；其他会话返回首页。
  useEffect(() => {
    let cancelled = false
    const request = ++detailRequestRef.current
    const generation = getStreamGeneration()
    const isCurrent = () => !cancelled && request === detailRequestRef.current
      && generation === getStreamGeneration()
    getSessionDetail(sessionId)
      .then((d) => {
        if (isCurrent()) setDetail(d)
      })
      .catch((e) => {
        if (!isCurrent()) return
        const knownDraft = e instanceof ApiError && e.status === 404
          && draftIdsRef.current?.has(sessionId)
        if (!knownDraft) navigate('/', { replace: true })
      })
    return () => {
      cancelled = true
    }
  }, [sessionId, navigate, draftIdsRef, getStreamGeneration])

  /* 新请求或新流开始后丢弃旧回放。 */
  const refreshDetail = useCallback(async (completion?: StreamCompletion) => {
    const generation = completion?.generation ?? getStreamGeneration()
    if (generation !== getStreamGeneration()) return
    const request = ++detailRequestRef.current
    try {
      const next = await getSessionDetail(sessionId)
      if (request !== detailRequestRef.current || generation !== getStreamGeneration()) return
      setDetail(next)
      // 运行中只同步审批；终态才替换正文。
      const current = next.turns.find((turn) => turn.turn_id === completion?.turnId)
      if (completion && current) reconcileTurn(completion, current, pendingFromDetail(next))
    } catch {
      /* 请求失败保留已显示内容。 */
    }
  }, [sessionId, reconcileTurn, getStreamGeneration])

  // 断线不取消后端执行；串行轮询至终态，旧 GET 仍校验身份。
  const needsReplayPolling = !isStreaming && (liveTurn?.state === 'running'
    || detail?.turns.some((turn) => turn.state === 'running'))
  useEffect(() => {
    if (!needsReplayPolling) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      await refreshDetail(getCompletion())
      if (!cancelled) timer = setTimeout(() => void poll(), 1000)
    }
    timer = setTimeout(() => void poll(), 1000)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [needsReplayPolling, refreshDetail, getCompletion])

  const refreshAfterTurn = useCallback((completion: StreamCompletion) => {
    void refreshDetail(completion)
    refreshPermission()
  }, [refreshDetail, refreshPermission])

  /* 活跃流拥有审批状态，禁止回放覆盖。 */
  const refreshAfterApproval = useCallback(() => {
    refreshPermission()
    // 审批响应可能晚于流结束，按当前身份收尾。
    if (!hasActiveStream()) void refreshDetail(getCompletion())
  }, [refreshPermission, hasActiveStream, refreshDetail, getCompletion])

  const handleSend = useCallback(
    (text: string, execution: ExecutionDraft, onAccepted?: () => void) => {
      void send(text, refreshAfterTurn, execution, onAccepted)
    },
    [send, refreshAfterTurn],
  )

  const handleStop = useCallback(() => {
    void stopTurn(sessionId)
  }, [sessionId])

  const handleRetry = useCallback(() => {
    // 重跑保留原用户消息；回放失败时用实时快照恢复。
    void regenerate(liveTurn ?? detail?.turns.at(-1) ?? null, refreshAfterTurn)
  }, [regenerate, refreshAfterTurn, liveTurn, detail])

  return (
    <>
      <ChatArea
        detail={detail}
        liveTurn={liveTurn}
        replacedTurnId={replacedTurnId}
        isStreaming={isStreaming}
        currentId={sessionId}
        streamError={streamError}
        livePendingApproval={pendingApproval}
        permission={permission}
        onSend={handleSend}
        onStop={handleStop}
        onRegenerate={handleRetry}
        onApprovalResolved={refreshAfterApproval}
        onOpenSubtask={setActiveSubtask}
      />

      {activeSubtask && (
        <SubtaskViewer subtask={activeSubtask} onClose={() => setActiveSubtask(null)} />
      )}
    </>
  )
}
