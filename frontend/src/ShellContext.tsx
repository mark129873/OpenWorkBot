import { createContext, useContext } from 'react'

export interface ShellState {
  sidebarOpen: boolean
  toggleSidebar: () => void
}

export const ShellContext = createContext<ShellState>({
  sidebarOpen: true,
  toggleSidebar: () => {},
})

export function useShell(): ShellState {
  return useContext(ShellContext)
}
