# Claude Agent SDK Configuration Reference

## Table of Contents

1. [ClaudeAgentOptions Parameters](#claudeagentoptions-parameters)
   - [Core Configuration](#core-configuration)
   - [Tool Control](#tool-control)
   - [Session Management](#session-management)
   - [Advanced Features](#advanced-features)
2. [Environment Variables](#environment-variables)
3. [Settings.json Structure](#settingsjson-structure)
4. [CLAUDE.md Loading](#claudemd-loading)
5. [Permission Rule Format](#permission-rule-format)
6. [Configuration Best Practices](#configuration-best-practices)
7. [Memory and Context Tips](#memory-and-context-tips)

## ClaudeAgentOptions Parameters

### Core Configuration

#### system_prompt
- **Type**: string
- **Purpose**: Replace default system prompt entirely
- **Example**: `"You are a specialized legal assistant"`

#### append_system_prompt
- **Type**: string  
- **Purpose**: Append to default system prompt
- **Example**: `"Focus on contract review and analysis"`

#### model
- **Type**: string
- **Default**: `"claude-sonnet-4-5"`
- **Options**: Various Claude model versions

#### max_turns
- **Type**: integer | None
- **Purpose**: Maximum conversation turns
- **Default**: None (unlimited)
- **Production tip**: Always set a limit to control costs

#### cwd
- **Type**: string
- **Purpose**: Working directory path
- **Default**: Current directory

### Tool Control

#### allowed_tools
- **Type**: list[string]
- **Purpose**: Explicitly permitted tools
- **Examples**: 
  - Read-only: `["Read", "Grep"]`
  - Development: `["Read", "Write", "Bash"]`
  - With MCP: `["mcp__server__tool", "Read"]`

#### disallowed_tools
- **Type**: list[string]
- **Purpose**: Explicitly blocked tools
- **Example**: `["Bash", "Delete", "WebSearch"]`

#### permission_mode
- **Type**: PermissionMode enum (SDK v2.0+)
- **Valid Options**:
  - `"acceptEdits"` - Auto-approve file edits only (recommended for production)
  - `"bypassPermissions"` - Skip all permission prompts (development/testing only)
  - `"default"` - SDK default behavior (standard permission prompts)
  - `"plan"` - Planning mode for specialized workflows
- **INVALID**: `"manual"` does NOT exist in SDK v2.0+ and will cause errors
- **INVALID**: `"acceptAll"` does NOT exist in SDK v2.0+

#### mcp_servers
- **Type**: dict
- **Purpose**: MCP server configurations
- **Format**:
```python
{
    "internal": sdk_server_object,  # In-process SDK server
    "external": {                    # External stdio server
        "type": "stdio",
        "command": "python",
        "args": ["-m", "server"]
    }
}
```

### Session Management

#### continue_conversation
- **Type**: boolean
- **Purpose**: Resume most recent session
- **Default**: False

#### resume
- **Type**: string
- **Purpose**: Session ID to resume
- **Example**: `"session-abc123"`

#### fork_session
- **Type**: boolean
- **Purpose**: Branch from resumed session
- **Default**: False

### Advanced Features

#### hooks
- **Type**: dict[HookEvent, list[HookMatcher]]
- **Purpose**: Lifecycle hooks for validation/audit
- **Example**:
```python
{
    "PreToolUse": [
        HookMatcher(matcher="Bash", hooks=[safety_hook])
    ],
    "PostToolUse": [
        HookMatcher(matcher="*", hooks=[logger_hook])
    ]
}
```

#### agents
- **Type**: dict[str, AgentDefinition]
- **Purpose**: Define subagents programmatically
- **Example**:
```python
{
    "analyzer": AgentDefinition(
        description="Code analysis agent",
        tools=["Read", "Grep"],
        prompt="Analyze code quality",
        model="claude-sonnet-4-5"
    )
}
```

#### setting_sources
- **Type**: list[string]
- **Options**: `["user", "project", "local"]`
- **Purpose**: Which filesystem settings to load
- **CRITICAL**: Setting to `["project"]` is REQUIRED for CLAUDE.md loading
- **Default**: `[]` (no settings loaded)
- **Production tip**: Use `[]` to avoid loading user settings in production

#### max_thinking_tokens
- **Type**: integer
- **Default**: 8000
- **Purpose**: Control reasoning capacity

#### add_dirs
- **Type**: list[string]
- **Purpose**: Additional accessible directories

#### extra_args
- **Type**: dict
- **Purpose**: Additional CLI arguments
- **Example**: `{"skill_ids": ["contract-analysis"]}`

## Environment Variables

### Authentication
- `ANTHROPIC_API_KEY` - Primary authentication
- `CLAUDE_CODE_USE_BEDROCK=1` - Use AWS Bedrock
- `CLAUDE_CODE_USE_VERTEX=1` - Use Google Vertex AI

### Configuration
- `CLAUDE_CODE_ENABLE_TELEMETRY` - Control telemetry

## Settings.json Structure

### Location Priority
1. `.claude/settings.json` (project)
2. `.claude/settings.local.json` (local overrides)
3. `~/.claude/settings.json` (user global)

### Schema
```json
{
  "permissions": {
    "allow": ["Read(./src/**)"],
    "deny": ["Bash(rm -rf)"],
    "ask": ["Write"]
  },
  "env": {
    "NODE_ENV": "development"
  },
  "model": "claude-sonnet-4-5",
  "hooks": {},
  "ui": {
    "theme": "dark"
  }
}
```

## CLAUDE.md Loading

### Hierarchy
1. User-level: `~/.claude/CLAUDE.md`
2. Project-level: `./CLAUDE.md` or `./.claude/CLAUDE.md`
3. Module-level: In subdirectories

### Enable Loading (REQUIRED)
```python
# CRITICAL: This is NOT optional - you MUST set setting_sources
ClaudeAgentOptions(
    setting_sources=["project"]  # REQUIRED for CLAUDE.md loading
)

# Without this, CLAUDE.md files will NOT be loaded
# Default is [] which loads nothing
```

### Selective Loading
```python
# Load only project settings (recommended)
setting_sources=["project"]

# Load everything (use with caution)
setting_sources=["user", "project", "local"]

# Load nothing (production)
setting_sources=[]
```

## Permission Rule Format

### Basic Syntax
- Tool only: `"Read"`
- With specifics: `"Bash(npm test)"`
- With globs: `"Read(./src/**)"`

### Examples
```json
{
  "allow": [
    "Read",                    // Allow all Read operations
    "Write(./outputs/**)",     // Write only to outputs
    "Bash(npm run test)"       // Specific bash command
  ],
  "deny": [
    "Delete",                  // Block all deletes
    "Bash(sudo *)",           // Block sudo commands
    "Read(./secrets/**)"      // Block secrets directory
  ]
}
```

## Configuration Best Practices

### Development
```python
ClaudeAgentOptions(
    permission_mode="bypassPermissions",  # SDK v2.0+: Skip all prompts for rapid dev
    allowed_tools=["Read", "Write", "Bash", "Grep"],
    max_turns=20,
    setting_sources=["project", "user"]
)
```

### Staging
```python
ClaudeAgentOptions(
    permission_mode="acceptEdits",
    allowed_tools=["Read", "Write", "Bash"],
    disallowed_tools=["WebSearch"],
    max_turns=10,
    hooks={"PreToolUse": [safety_hooks]}
)
```

### Production
```python
ClaudeAgentOptions(
    permission_mode="acceptEdits",  # SDK v2.0+: Auto-accept with safety hooks
    allowed_tools=["Read", "Grep"],
    disallowed_tools=["Bash", "Write", "Delete"],
    max_turns=5,
    setting_sources=[],  # No user settings
    hooks={
        "PreToolUse": [comprehensive_safety_hooks],
        "PostToolUse": [audit_hooks]
    }
)
```

## Memory and Context Tips

### Token Optimization
1. Use `setting_sources` selectively
2. Structure CLAUDE.md with essentials first
3. Remove stale information regularly
4. Use retrieval over inclusion

### Context Compaction
- Automatic at ~95% capacity
- Manual via `/compact` command
- Preserves key information
- 84% token reduction (Anthropic testing)

### Caching Benefits
- System prompts cached automatically
- CLAUDE.md content cached
- Tool definitions cached
- 85% latency reduction possible
- 90% cost reduction for cached content
