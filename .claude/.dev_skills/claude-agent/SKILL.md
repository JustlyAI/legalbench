---
name: claude-agent-sdk
description: Expert guidance for building sophisticated AI agents using Claude Agent SDK. Provides patterns for production deployment, custom tool creation, safety controls, session management, memory handling, and Claude Agent Skills integration. Essential for developing agents with complex workflows, multi-agent orchestration, and enterprise requirements.
---

# Claude Agent SDK Expert

This skill provides expert guidance on the Claude Agent SDK for building production-ready AI agents with advanced capabilities including custom tools, safety hooks, session persistence, and integration patterns.

## Core Capabilities

The Claude Agent SDK enables sophisticated AI agent development through:

- **Dual Interaction Modes**: Stateless `query()` for one-shot tasks, stateful `ClaudeSDKClient` for conversations
- **Custom Tool Creation**: In-process MCP servers with zero subprocess overhead
- **Safety Controls**: Pre/Post hooks for validation, rate limiting, and audit logging
- **Session Management**: Resume and fork capabilities with full conversation history
- **Memory Systems**: CLAUDE.md loading and automatic prompt caching
- **Skills Integration**: Progressive-loading capabilities without token penalties
- **Production Patterns**: Error handling, retry logic, and orchestrator architectures

## When to Use This Skill

Activate this skill when:

- Setting up a new Claude Agent SDK project
- Creating custom tools or MCP servers
- Implementing safety hooks and permission controls
- Managing sessions and conversation persistence
- Optimizing token usage and costs
- Integrating Claude Agent Skills
- Troubleshooting SDK issues
- Building multi-agent systems
- Deploying agents to production

## Available Resources

The github repository for the claude agent sdk is https://github.com/anthropics/claude-agent-sdk-python

### Scripts

- `scripts/quick_start.py`: Interactive examples for query(), ClaudeSDKClient, and session management
- `scripts/custom_tools.py`: Templates for creating tools with validation, MCP servers, and hybrid configurations
- `scripts/production_patterns.py`: Safety hooks, error handling, orchestrator pattern, and monitoring

### References

- `references/api_patterns.md`: Core interaction patterns, tool management, message types, and optimization strategies
- `references/configuration.md`: Complete ClaudeAgentOptions parameters, environment variables, and best practices
- `references/skills_integration.md`: Claude Agent Skills architecture, SDK integration, migration strategies

## Key Patterns

### Basic Agent Setup

```python
from claude_agent_sdk import query, ClaudeAgentOptions

async for message in query(
    prompt="Your task here",
    options=ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        max_turns=5,
        allowed_tools=["Read", "Write"],
        setting_sources=["project"]  # REQUIRED for CLAUDE.md loading
    )
):
    # Process messages
```

### Session ID Capture (Critical Pattern)

```python
from claude_agent_sdk import SystemMessage

session_id = None
async for message in query(prompt="Task", options=options):
    if isinstance(message, SystemMessage) and message.subtype == 'init':
        session_id = UUID(message.data.get('session_id'))
        # Store session_id for later resumption
```

### Custom Tool Creation

```python
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool("tool_name", "description", {"param": str})
async def custom_tool(args):
    return {"content": [{"type": "text", "text": "result"}]}

server = create_sdk_mcp_server(
    name="tools",
    version="1.0.0",
    tools=[custom_tool]
)
```

### Production Configuration

```python
ClaudeAgentOptions(
    permission_mode="acceptEdits",  # SDK v2.0+: Auto-accept file modifications
    allowed_tools=["Read"],         # Minimal permissions
    hooks={
        "PreToolUse": [HookMatcher(matcher="*", hooks=[safety_hook])],
        "PostToolUse": [HookMatcher(matcher="*", hooks=[audit_hook])]
    },
    max_turns=5,  # Limit iterations
    setting_sources=[]  # No user settings in production
)
```

## Architecture Decisions

### Query vs Client

- **Use query()**: One-shot tasks, batch processing, CLI tools
- **Use ClaudeSDKClient**: Conversations, complex workflows, session management

### Tools vs Skills

- **SDK Tools**: Real-time operations, external APIs, dynamic data
- **Skills**: Procedural workflows, validation, document templates

### Permission Modes (SDK v2.0+)

- **Development**: `bypassPermissions` for rapid iteration (no prompts)
- **Staging**: `acceptEdits` for balanced safety (auto-approve file edits)
- **Production**: `acceptEdits` with safety hooks for controlled automation
- **Note**: `manual` mode does NOT exist in SDK v2.0+ - use `default` for standard prompts

## Critical SDK Patterns

### Message Storage Pattern

```python
# ALWAYS store messages immediately (no buffering for durability)
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
    logger.info(f"Cache read: {usage.get('cache_read_input_tokens', 0)}")
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

## Best Practices

1. **Start with read-only**: Begin with `["Read", "Grep"]`, add permissions gradually
2. **Implement safety early**: Add hooks before production deployment
3. **Monitor everything**: Track costs via ResultMessage with Decimal precision
4. **Use progressive loading**: Skills and CLAUDE.md for context efficiency
5. **Test thoroughly**: Unit test tools, integration test workflows
6. **Handle errors gracefully**: Implement retry with exponential backoff
7. **Version control configurations**: Track agent settings with Git
8. **Store messages immediately**: No buffering - write to database on receipt
9. **Extract session ID early**: Capture from SystemMessage.subtype == 'init'
10. **Use async context managers**: Ensure proper connection cleanup

## Common Issues & Solutions

### CLI Not Found

```python
except CLINotFoundError:
    print("Install: npm install -g @anthropic-ai/claude-code")
```

### Connection Errors

Implement retry logic with exponential backoff - see `scripts/production_patterns.py`

### Token Optimization

- Set `max_turns` limits
- Use `setting_sources=["project"]` selectively
- Implement Skills for large reference materials

### Session Management

Always capture session_id from init messages for later resumption

## Advanced Techniques

### Orchestrator Pattern

Create specialized agents for different tasks, coordinate via orchestrator:

- Analyzer agent (read-only)
- Implementer agent (write access)
- Tester agent (execution)

### Hybrid Tool Approach

Combine in-process SDK servers (fast, trusted) with external MCP servers (isolated, third-party)

### Skills Integration

Use Skills for procedural knowledge, SDK tools for real-time operations

## Quick Command Reference

Run examples:

```bash
python scripts/quick_start.py      # Interactive examples
python scripts/custom_tools.py     # Tool creation demo
python scripts/production_patterns.py  # Production patterns
```

Check references for:

- API patterns: Tool management, session handling, optimization
- Configuration: All ClaudeAgentOptions parameters
- Skills: Integration patterns and migration strategies
