# Claude Agent SDK API Patterns Reference

## Table of Contents

1. [Core Interaction Patterns](#core-interaction-patterns)
2. [Tool Management Patterns](#tool-management-patterns)
3. [Session Management](#session-management)
4. [Error Handling](#error-handling)
5. [Message Types](#message-types)
6. [Performance Optimization](#performance-optimization)
7. [Safety Patterns](#safety-patterns)
8. [Parallel Processing](#parallel-processing)
9. [Configuration Priorities](#configuration-priorities)
10. [Database Integration Patterns](#database-integration-patterns)
11. [Best Practices Summary](#best-practices-summary)

## Core Interaction Patterns

### Stateless Query Pattern
Use `query()` for one-shot operations without conversation history:
- Simple automation scripts
- Batch processing
- Independent tasks
- CLI tools

**Best for**: Tasks that don't require context between calls

**Example**:
```python
from claude_agent_sdk import query, ClaudeAgentOptions

async for message in query(
    prompt="Analyze this file",
    options=ClaudeAgentOptions(allowed_tools=["Read"])
):
    # Process messages
    pass
```

### Stateful Client Pattern
Use `ClaudeSDKClient` for multi-turn conversations:
- Interactive applications
- Complex workflows requiring context
- Response-driven logic
- Session management

**Best for**: Applications where Claude's responses inform subsequent queries

**Example**:
```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

async with ClaudeSDKClient(options=options) as client:
    await client.query("First question")
    async for msg in client.receive_response():
        # Process first response
        pass

    await client.query("Follow-up question")
    async for msg in client.receive_response():
        # Process follow-up (maintains context)
        pass
```

## Tool Management Patterns

### Built-in Tools
```python
allowed_tools = ["Read", "Write", "Bash", "Grep", "WebSearch", "WebFetch"]
```

### Custom Tool Pattern
```python
@tool("name", "description", {"param": type})
async def tool_name(args):
    return {"content": [{"type": "text", "text": "result"}]}
```

### MCP Server Naming
Tools in MCP servers follow: `mcp__{server_name}__{tool_name}`

## Session Management

### Getting Session ID (CRITICAL PATTERN)
```python
from claude_agent_sdk import SystemMessage
from uuid import UUID

session_id = None
async for message in query(prompt="...", options=options):
    # CORRECT: Check type first, then subtype
    if isinstance(message, SystemMessage) and message.subtype == 'init':
        session_id = UUID(message.data.get('session_id'))
        # Store session_id for later resumption
```

### Resuming Sessions
```python
# Continue existing session
options = ClaudeAgentOptions(resume=str(session_id))

# Or fork to create new branch from session
options = ClaudeAgentOptions(
    resume=str(session_id),
    fork_session=True  # Creates new session ID
)
```

### Dual Storage Architecture
```python
# PostgreSQL mode: Full conversation history
try:
    session_manager = SessionManager(database_url)
    await session_manager.initialize()
except Exception as e:
    logger.warning(f"PostgreSQL unavailable: {e}")
    session_manager = None  # Fall back to JSON mode

# Both modes support SDK session resumption
if session_manager:
    await session_manager.create_session(session_id, config_data)
else:
    save_session_to_json(session_id)
```

## Error Handling

### Exception Types
- `CLINotFoundError` - Claude CLI not installed
- `CLIConnectionError` - Connection issues
- `ProcessError` - Process failures (includes exit_code)
- `CLIJSONDecodeError` - JSON parsing errors

### Retry Strategy
```python
for attempt in range(max_retries):
    try:
        # Your query
    except (CLIConnectionError, ProcessError) as e:
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        else:
            raise
```

## Message Types

### Input Messages
- `UserMessage` - User input
- `SystemMessage` - System metadata

### Output Messages  
- `AssistantMessage` - Claude's response with content blocks
- `ResultMessage` - Final result with cost and usage data

### Content Blocks
- `TextBlock` - Natural language text
- `ThinkingBlock` - Reasoning process (extended thinking models)
- `ToolUseBlock` - Tool invocation details
- `ToolResultBlock` - Tool execution results

## Performance Optimization

### Context Management
- Use `setting_sources=["project"]` to load only needed CLAUDE.md files
- Let automatic compaction handle long conversations
- Use retrieval tools (Grep) over loading full file contents

### Token Control
- Set `max_turns` to limit iterations
- Use `max_thinking_tokens` to control reasoning
- Structure CLAUDE.md files with essential info upfront

### Caching Strategy
- Place static content at beginning of system prompts
- Use consistent prompts across requests
- Make requests within cache TTL window (5 min standard, 1 hour extended)

## Safety Patterns

### Permission Modes (SDK v2.0+)
- `acceptEdits` - Auto-approve file edits only (recommended for production)
- `bypassPermissions` - Skip all permission prompts (development/testing only)
- `default` - SDK default behavior with standard prompts
- `plan` - Planning mode for specialized workflows
- **INVALID**: `manual` does NOT exist in SDK v2.0+ - use `acceptEdits` with hooks instead
- **INVALID**: `acceptAll` does NOT exist in SDK v2.0+

### Hook Types
- `PreToolUse` - Validate before tool execution
- `PostToolUse` - Audit after tool completion

## Parallel Processing

### Batch Operations
```python
async def batch_process(tasks):
    results = await asyncio.gather(*[
        query(prompt=task) for task in tasks
    ])
    return results
```

### Task Groups (anyio)
```python
async with anyio.create_task_group() as tg:
    tg.start_soon(query_task_1)
    tg.start_soon(query_task_2)
```

## Configuration Priorities

1. Explicit parameters in ClaudeAgentOptions
2. Environment variables
3. Settings.json files (project > local > user)
4. Default values

## Database Integration Patterns

### Immediate Message Storage
```python
# CRITICAL: Store messages immediately (no buffering for durability)
async for message in client.receive_response():
    # 1. Create session on init
    if isinstance(message, SystemMessage) and message.subtype == 'init':
        session_id = UUID(message.data.get('session_id'))
        await session_manager.create_session(session_id, message.data, matter_id)

    # 2. Store immediately
    await session_manager.store_message(message, session_id)

    # 3. Finalize on completion
    if isinstance(message, ResultMessage):
        await session_manager.finalize_session(session_id, message)
```

### Cost Tracking with Precision
```python
from decimal import Decimal

# ALWAYS convert via string to preserve precision
if isinstance(message, ResultMessage):
    cost = Decimal(str(message.total_cost_usd)) if message.total_cost_usd else None
    usage = message.usage or {}

    logger.info(f"Cost: ${cost:.6f}, Tokens: {usage.get('total_tokens', 0)}")
```

### Manual UserMessage Creation
```python
# SDK doesn't echo user messages - create manually for storage
from claude_agent_sdk import UserMessage

user_input = input("You: ")
user_message = UserMessage(content=user_input)
await session_manager.store_message(user_message, session_id)
await client.query(user_input)
```

## Best Practices Summary

1. **Start restrictive**: Begin with read-only tools, add permissions gradually
2. **Use appropriate mode**: Query for stateless, Client for conversations
3. **Implement safety hooks**: Always in production environments
4. **Monitor costs**: Track via ResultMessage with Decimal precision
5. **Handle errors gracefully**: Implement retry logic with backoff
6. **Optimize context**: Use progressive loading and retrieval patterns
7. **Version control Skills**: Track changes with Git
8. **Test thoroughly**: Unit test tools, integration test workflows
9. **Document patterns**: Maintain clear documentation of agent behaviors
10. **Audit everything**: Log tool usage, decisions, and costs
11. **Store messages immediately**: No buffering - write to database on receipt
12. **Extract session ID correctly**: Use isinstance() and type check first
13. **Use async context managers**: Ensure proper connection cleanup
14. **Create UserMessages manually**: SDK doesn't echo user input
