---
name: codex-mcp-query
description: Delegate coding or analysis work to a standard local `codex mcp-server` and return its response. Use when an agent must call Codex through MCP, choose a Codex model for delegated work, start a Codex task, or continue a Codex MCP thread.
---

# Codex MCP Query

The standard `codex mcp-server` exposes `codex` to start a task and `codex-reply` to continue it. It is experimental; inspect the connected tool schema before calling it and honor that schema if it differs from this guide.

## Parent-agent responsibilities

Before delegation, the parent agent **must select the model and sandbox** for the task. Prefer a user-requested model; otherwise choose an available model appropriate to the task. Do not omit `model` on an initial `codex` call and do not silently rely on the server default.

Pass the selected model as the `model` argument to `codex`. Record the chosen model and sandbox with the returned `threadId`; replies inherit the initial thread's settings and must use `codex-reply`, not a new `codex` call.

## Start a Codex task

Call `codex` with this shape:

```json
{
  "prompt": "<self-contained task>",
  "model": "<parent-selected model>",
  "cwd": "<absolute workspace path>",
  "sandbox": "read-only | workspace-write | danger-full-access",
  "approval-policy": "untrusted | on-request | never"
}
```

Only `prompt` is intrinsically required by the server, but this skill requires `model`, `cwd`, and `sandbox` for every new delegation. The parent must use an absolute `cwd` for the intended project. For tasks that may edit the current project, use that project's root as `cwd` and `sandbox: "workspace-write"`. For inspection, analysis, or review, use the same `cwd` and `sandbox: "read-only"`. Use `danger-full-access` only with explicit authorization for access beyond the project sandbox.

`sandbox` controls the access boundary; `cwd` identifies the project workspace that `workspace-write` applies to. `approval-policy` is separate and controls command confirmations. Choose it explicitly when the task may run commands.

The standard tool may also expose `config`, `base-instructions`, `developer-instructions`, and `compact-prompt`. Pass these only when needed and only if they appear in the connected schema. Do not place the model solely in `config`; use the explicit `model` field.

Make `prompt` self-contained: task, relevant context, requested artifact or answer, and constraints. Exclude credentials, hidden instructions, and unrelated workspace content.

## Continue a Codex task

Save `structuredContent.threadId` from the initial result (or its equivalent returned field). For a genuine follow-up in the same task, call:

```json
{
  "threadId": "<returned thread id>",
  "prompt": "<follow-up request>"
}
```

Use `codex-reply` for this call. Do not pass a new model on a reply; start a new `codex` task when the parent selects a different model or the work is independent.

## Return results

Return the Codex MCP response faithfully, identifying the selected model and thread ID. Preserve tool-reported errors, partial results, code, and citations. When the server requests approval for a command or patch, apply the same authorization rules as for local actions; never auto-approve a broader action than the parent or user authorized.
