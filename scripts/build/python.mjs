#!/usr/bin/env node
/** Run build helpers with the prepared Python, never an ambient uv environment. */
import path from 'node:path'
import os from 'node:os'
import { execFileSync } from 'node:child_process'
import { existsSync, readdirSync, statSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

/**
 * The prepared build interpreter: $HERMES_PYTHON when set, otherwise the newest
 * one the installer left under $HERMES_HOME/tools. A bundling run is long enough
 * that failing here on an unset variable — after `before-pack` has already wiped
 * release/ — costs a whole rebuild.
 * @param {NodeJS.ProcessEnv} env
 */
export function resolveHermesPython(env) {
  if (env.HERMES_PYTHON) return env.HERMES_PYTHON
  const home = env.HERMES_HOME || path.join(os.homedir(), '.hermes')
  const tools = path.join(home, 'tools')
  const candidates = existsSync(tools)
    ? readdirSync(tools)
        .filter((name) => name.startsWith('python-'))
        .map((name) => path.join(tools, name, 'bin', 'python3'))
        .filter((candidate) => existsSync(candidate))
        .sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs)
    : []
  if (candidates.length) return candidates[0]
  throw new Error(
    'No prepared build interpreter: set HERMES_PYTHON, or install one at ' +
      `${path.join(tools, 'python-<version>', 'bin', 'python3')}`,
  )
}

/**
 * @param {string[]} args
 * @param {import('node:child_process').ExecFileSyncOptions} [options]
 */
export function runPython(args, options = {}) {
  const env = options.env ?? process.env
  const python = resolveHermesPython(env)
  // Hand the resolved path down: nested build helpers read the same variable.
  return execFileSync(python, args, { stdio: 'inherit', ...options, env: { ...env, HERMES_PYTHON: python } })
}


if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    runPython(process.argv.slice(2))
  } catch (/** @type {unknown} */ error) {
    // execFileSync throws a child_process exception carrying .status; anything
    // else is reported as-is so a missing Python still prints a useful line.
    const detail = error instanceof Error ? error.message : String(error)
    console.error('[python build]', detail)
    const status = typeof (/** @type {any} */ (error).status) === 'number' ? /** @type {any} */ (error).status : 0
    process.exitCode = status !== 0 ? status : 1
  }
}
