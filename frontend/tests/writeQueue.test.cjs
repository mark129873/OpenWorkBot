const { test } = require('node:test')
const assert = require('node:assert/strict')
const { createWriteQueue } = require('../../tmp/frontend-tests/utils/writeQueue.js')

test('messages and approvals are submitted in order', async () => {
  const enqueue = createWriteQueue()
  const calls = []
  let release
  const gate = new Promise((resolve) => { release = resolve })
  const first = enqueue(async () => { calls.push('message'); await gate })
  const second = enqueue(async () => { calls.push('approval') })
  await Promise.resolve()
  assert.deepEqual(calls, ['message'])
  release()
  await Promise.all([first, second])
  assert.deepEqual(calls, ['message', 'approval'])
})

test('a failed submission prevents later queued actions from overtaking it', async () => {
  const enqueue = createWriteQueue()
  const calls = []
  const first = enqueue(async () => { calls.push('message'); throw new Error('offline') })
  const second = enqueue(async () => { calls.push('approval') })
  await assert.rejects(first, /offline/)
  await assert.rejects(second, /offline/)
  assert.deepEqual(calls, ['message'])
  await enqueue(async () => { calls.push('retry') })
  assert.deepEqual(calls, ['message', 'retry'])
})
