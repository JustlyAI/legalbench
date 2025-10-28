# Claude Agent Skills Integration Reference

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [SDK Integration Patterns](#sdk-integration-patterns)
4. [Building Skills for Agents](#building-skills-for-agents)
5. [When to Use Skills vs SDK Tools](#when-to-use-skills-vs-sdk-tools)
6. [Pre-Built Skills](#pre-built-skills)
7. [Multi-Skill Composition](#multi-skill-composition)
8. [Migration from Memory Tool](#migration-from-memory-tool)
9. [Best Practices](#best-practices)
10. [Performance Metrics](#performance-metrics)
11. [Security Considerations](#security-considerations)
12. [Debugging Skills](#debugging-skills)

## Overview

Claude Agent Skills extend the SDK with filesystem-based, progressive-loading capabilities that transform general-purpose Claude into domain specialists without context window penalties.

## Architecture

### Three-Tier Loading Model

| Level | Content | Token Cost | When Loaded |
|-------|---------|------------|-------------|
| **Level 1** | Metadata (name + description) | ~100 tokens | At startup |
| **Level 2** | SKILL.md body | <5k tokens | When triggered |
| **Level 3** | Resources (scripts/refs/assets) | Unlimited* | As needed |

*Scripts execute without loading into context

### Key Innovation
Scripts in Skills execute without their code entering the context window—only their output does. This enables bundling comprehensive resources without token penalties.

## SDK Integration Patterns

### Pattern 1: Claude Code CLI with Skills
Claude Agent SDK integrates with Claude Code CLI, which has built-in Skills support:

```python
from claude_agent_sdk import query, ClaudeAgentOptions

# Claude Code CLI automatically discovers skills at:
# - ~/.claude/skills/ (user skills)
# - ./.claude/skills/ (project skills)

# SDK queries automatically have access to Skills
async for message in query(
    prompt="Analyze this contract using legal procedures",
    options=ClaudeAgentOptions(
        setting_sources=["project"],  # Load project CLAUDE.md
        allowed_tools=["Read", "Write"]
    )
):
    # Skills are activated automatically based on prompt
    pass
```

### Pattern 2: Custom Project Skills Structure
```bash
# Project skills location
.claude/skills/
├── legal-analysis/
│   ├── SKILL.md          # Required
│   ├── scripts/          # Optional: executable code
│   │   └── validator.py
│   └── references/       # Optional: documentation
│       └── procedures.md
└── document-creation/
    ├── SKILL.md
    └── templates/
        └── memo.docx
```

### Pattern 3: Hybrid SDK Tools + Skills
```python
from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions

# Real-time operations via SDK tools (database, APIs)
@tool("live_query", "Query database", {"query": str})
async def live_query(args):
    return await db.query(args["query"])

server = create_sdk_mcp_server(
    name="tools",
    version="1.0.0",
    tools=[live_query]
)

# Procedural operations via Skills (.claude/skills/)
# Skills provide workflows, validation, templates

options = ClaudeAgentOptions(
    mcp_servers={"tools": server},      # Real-time SDK tools
    setting_sources=["project"],        # Enable Skills discovery
    allowed_tools=["mcp__tools__live_query", "Read"]
)

# Claude automatically uses:
# - SDK tools for real-time data operations
# - Skills for procedural workflows and validation
```

### Pattern 4: Skills for Reference Documentation
```python
# Instead of loading large docs into system prompt:
# BAD: Large system prompt with all documentation
system_prompt = load_file("legal_procedures.md")  # 10,000 tokens

# GOOD: Use Skills for progressive loading
# .claude/skills/legal-procedures/SKILL.md
# - Level 1: ~100 tokens (metadata)
# - Level 2: ~2,000 tokens (instructions, loaded when triggered)
# - Level 3: Unlimited (scripts/references, loaded as needed)

options = ClaudeAgentOptions(
    setting_sources=["project"],  # Enable Skills
    # System prompt stays small
)
```

## Building Skills for Agents

### Skill Structure
```
skill-name/
├── SKILL.md              # Required: metadata + instructions
├── scripts/              # Optional: executable code
│   └── validator.py
├── references/           # Optional: documentation
│   └── patterns.md
└── assets/               # Optional: output files
    └── template.docx
```

### SKILL.md Format
```markdown
---
name: skill-name           # Max 64 chars
description: Clear description of functionality and triggers  # Max 1024 chars
---

# Instructions

Core procedures here (keep under 500 lines)

## Resources

- Scripts: `scripts/process.py` for validation
- References: See `references/details.md` for specifications
- Assets: Use `assets/template.docx` for output
```

### When to Use Skills vs SDK Tools

| Use Case | Skills | SDK Tools |
|----------|--------|-----------|
| Procedural workflows | ✓ Best | Use for dynamic |
| Validation/formatting | ✓ Best | Possible but verbose |
| Real-time data access | Limited | ✓ Best |
| External API calls | Not supported | ✓ Best |
| Document templates | ✓ Best | Requires generation |
| Reference documentation | ✓ Best | Token intensive |

## Pre-Built Skills

### Document Creation
- **pptx**: PowerPoint presentations
- **xlsx**: Excel spreadsheets with formulas
- **docx**: Word documents with formatting
- **pdf**: PDF creation and manipulation

### Integration with Claude Code CLI
```python
from claude_agent_sdk import query, ClaudeAgentOptions

# Claude Code CLI has built-in Skills like pptx, xlsx, docx, pdf
# These are available automatically when using Claude Code

options = ClaudeAgentOptions(
    setting_sources=["project"],  # Enable Skills discovery
    allowed_tools=["Read", "Write"]
)

# Claude automatically uses appropriate Skill based on task
async for message in query(
    prompt="Create a presentation about Q4 results",
    options=options
):
    # Claude Code CLI activates pptx skill automatically
    pass
```

## Multi-Skill Composition

### Sequential Workflow
```python
async def complex_workflow(case_id):
    response = await client.query(f"""
    For case {case_id}:
    1. Analyze contract (contract-analysis skill)
    2. Create risk presentation (pptx skill)  
    3. Generate financial analysis (xlsx skill)
    4. Draft legal memo (docx skill)
    """)
    # Each skill activates automatically
```

### Skill Coordination
Skills can reference each other's outputs when in the same session:
- First skill creates `analysis.json`
- Second skill reads and processes it
- Third skill generates final output

## Migration from Memory Tool

### Before (Memory Tool)
```python
# Everything loads into context
memories/
├── knowledge/legal_procedures.md  # 2000 tokens
├── knowledge/citations.md         # 1500 tokens
└── context/current_case.md        # 1000 tokens
Total: 4500 tokens loaded upfront
```

### After (Skills)
```python
# Progressive loading
skills/
├── legal-procedures/    # ~100 tokens metadata
│   └── SKILL.md        # Loads only when needed
└── citation-validator/  # ~100 tokens metadata
    └── scripts/validate.py  # Executes without loading
Total: ~200 tokens initially
```

## Best Practices

### DO's
- ✓ Keep SKILL.md under 500 lines
- ✓ Use scripts for deterministic operations
- ✓ Bundle comprehensive references
- ✓ Version control Skills with Git
- ✓ Test scripts independently
- ✓ Use descriptive skill names

### DON'Ts
- ✗ Don't duplicate info across Skills
- ✗ Don't hardcode secrets
- ✗ Don't exceed 5KB in SKILL.md body
- ✗ Don't create auxiliary docs (README, etc.)
- ✗ Don't nest references deeply

## Performance Metrics

### Token Savings
| Approach | Initial | Per Query | 100 Queries |
|----------|---------|-----------|-------------|
| Traditional | 6,500 | 6,500 | 650,000 |
| Memory Tool | 3,000 | 3,000 | 300,000 |
| Skills | 300 | 800 avg | 80,300 |

**Result**: 87% token reduction vs traditional, 73% vs Memory Tool

### Cost Impact
- Cache writes: 25% more than base tokens
- Cache reads: 10% of base token cost
- Skills leverage caching automatically
- Typical savings: $1,000+/month at scale

## Security Considerations

### Container Restrictions
```python
container_limits = {
    "network_access": False,      # No external calls
    "filesystem_access": "limited",  # Skill dir + workspace
    "package_installation": False,   # Pre-installed only
    "execution_timeout": 30         # Max seconds
}
```

### Validation Requirements
1. Review all Skill code before deployment
2. Check for malicious patterns
3. Validate external dependencies
4. Implement audit logging
5. Regular security reviews

## Debugging Skills

### Testing Scripts
```python
# Test skill scripts independently
python skills/legal-analysis/scripts/validator.py test_input.txt
```

### Monitoring Activation
```python
# Log skill usage
async def skill_monitor_hook(input_data, tool_use_id, context):
    if "skill_activated" in input_data:
        logger.info(f"Skill: {input_data['skill_name']}")
    return {}
```

### Common Issues
1. **Skill not triggering**: Check description specificity
2. **Script errors**: Test scripts standalone first
3. **Token overflow**: Move content to references
4. **Missing resources**: Verify file paths in SKILL.md
