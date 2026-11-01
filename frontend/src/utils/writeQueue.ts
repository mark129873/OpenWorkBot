export function createWriteQueue() {
  let tail: Promise<unknown> = Promise.resolve()
  return (action: () => Promise<unknown>): Promise<unknown> => {
    const result = tail.then(action)
    tail = result
    void result.catch(() => {
      if (tail === result) tail = Promise.resolve()
    })
    return result
  }
}
