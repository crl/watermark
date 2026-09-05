import { ChildProcess, spawn } from 'child_process'
import { existsSync } from 'fs'
import { join } from 'path'
import { app } from 'electron'

export type SidecarState = {
  status: 'starting' | 'ready' | 'error' | 'stopped'
  port: number | null
  error: string | null
  python: string | null
}

function backendRoot(): string {
  if (app.isPackaged) {
    return join(process.resourcesPath, 'backend')
  }
  return join(__dirname, '../../../backend')
}

function pythonCandidates(): string[] {
  const root = backendRoot()
  const list = [
    process.env.WATERMARK_PYTHON,
    join(root, '.venv', 'Scripts', 'python.exe'),
    join(root, '.venv', 'bin', 'python'),
    join(root, 'venv', 'Scripts', 'python.exe'),
    process.platform === 'win32' ? 'python' : 'python3',
    'py',
    'python'
  ]
  return list.filter((item): item is string => Boolean(item))
}

export class Sidecar {
  process: ChildProcess | null = null
  state: SidecarState = { status: 'starting', port: null, error: null, python: null }
  listeners = new Set<(state: SidecarState) => void>()

  onChange(listener: (state: SidecarState) => void): () => void {
    this.listeners.add(listener)
    listener(this.state)
    return () => this.listeners.delete(listener)
  }

  private emit(patch: Partial<SidecarState>): void {
    this.state = { ...this.state, ...patch }
    for (const listener of this.listeners) listener(this.state)
  }

  get apiBase(): string | null {
    return this.state.port ? `http://127.0.0.1:${this.state.port}` : null
  }

  async start(): Promise<string> {
    const root = backendRoot()
    const entry = join(root, 'app.py')
    if (!existsSync(entry)) {
      const message = `找不到后端：${entry}`
      this.emit({ status: 'error', error: message })
      throw new Error(message)
    }

    let lastError = ''
    for (const python of pythonCandidates()) {
      try {
        const port = await this.spawnPython(python, root, entry)
        this.emit({ status: 'ready', port, error: null, python })
        return `http://127.0.0.1:${port}`
      } catch (error) {
        lastError = error instanceof Error ? error.message : String(error)
        console.error(`[sidecar] ${python} failed:`, lastError)
      }
    }
    this.emit({ status: 'error', error: lastError || '无法启动 Python 后端' })
    throw new Error(lastError)
  }

  private spawnPython(python: string, cwd: string, entry: string): Promise<number> {
    return new Promise((resolve, reject) => {
      const args = python === 'py' ? ['-3', '-u', entry] : ['-u', entry]
      const child = spawn(python, args, {
        cwd,
        env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
        windowsHide: true
      })
      this.process = child
      let buffer = ''
      let settled = false

      const timer = setTimeout(() => {
        if (!settled) {
          settled = true
          child.kill()
          reject(new Error(`启动超时。请先运行 pip install -r backend/requirements.txt\n${buffer.slice(-1500)}`))
        }
      }, 45000)

      const onData = (chunk: Buffer): void => {
        const text = chunk.toString('utf8')
        buffer += text
        const match = text.match(/WATERMARK_READY port=(\d+)/)
        if (match && !settled) {
          settled = true
          clearTimeout(timer)
          console.log(`[sidecar] ready on ${match[1]} via ${python}`)
          resolve(Number(match[1]))
        }
      }

      child.stdout?.on('data', onData)
      child.stderr?.on('data', onData)
      child.once('error', (error) => {
        if (!settled) {
          settled = true
          clearTimeout(timer)
          reject(error)
        }
      })
      child.once('exit', (code) => {
        this.process = null
        if (!settled) {
          settled = true
          clearTimeout(timer)
          reject(new Error(`后端退出（${code}）\n${buffer.slice(-1500)}`))
        } else {
          this.emit({ status: 'stopped', port: null, error: '后端已退出' })
        }
      })
    })
  }

  stop(): void {
    if (this.process && !this.process.killed) {
      this.process.kill()
    }
    this.process = null
  }
}
