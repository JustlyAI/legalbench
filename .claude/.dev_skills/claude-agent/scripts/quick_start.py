#!/usr/bin/env python3
"""Quick start template for Claude Agent SDK projects"""

import asyncio
from pathlib import Path
from claude_agent_sdk import (
    query, ClaudeSDKClient, ClaudeAgentOptions,
    AssistantMessage, TextBlock, ToolUseBlock, ResultMessage
)

# Basic query example - stateless one-shot operation
async def simple_query_example():
    """Demonstrates basic query() usage for stateless operations"""
    async for message in query(
        prompt="Analyze this Python file and suggest improvements",
        options=ClaudeAgentOptions(
            model="claude-sonnet-4-5",
            max_turns=3,
            allowed_tools=["Read", "Grep"],
            cwd=".",
        )
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)
        elif isinstance(message, ResultMessage):
            print(f"\n\nCost: ${message.total_cost_usd:.4f}")
            break

# Stateful client example - conversation with context
async def conversation_example():
    """Demonstrates ClaudeSDKClient for multi-turn conversations"""
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        max_turns=10,
        allowed_tools=["Read", "Write", "Bash"],
        setting_sources=["project"],  # Load CLAUDE.md files
        permission_mode="acceptEdits",  # SDK v2.0+: Auto-accept file edits
        cwd=".",
    )
    
    async with ClaudeSDKClient(options=options) as client:
        # First query
        await client.query("List Python files in the current directory")
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)
        
        # Follow-up query remembers context
        print("\n\n--- Follow-up Query ---\n")
        await client.query("Now analyze the main.py file for improvements")
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)

# Session management example
async def session_management_example():
    """Demonstrates session persistence and resumption"""
    from claude_agent_sdk import SystemMessage
    from uuid import UUID

    session_id = None

    # Create initial session
    async for message in query(
        prompt="Help me design a REST API",
        options=ClaudeAgentOptions(model="claude-sonnet-4-5")
    ):
        # CORRECT pattern: Check type first, then subtype
        if isinstance(message, SystemMessage) and message.subtype == 'init':
            session_id = UUID(message.data.get('session_id'))
            print(f"Session started: {session_id}")
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)

    # Resume session later
    if session_id:
        print("\n\n--- Resuming Session ---\n")
        async for message in query(
            prompt="Continue with authentication implementation",
            options=ClaudeAgentOptions(
                resume=str(session_id),
                model="claude-sonnet-4-5"
            )
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)

if __name__ == "__main__":
    print("Claude Agent SDK Quick Start Examples\n")
    print("1. Simple Query (stateless)")
    print("2. Conversation (stateful)")
    print("3. Session Management")
    
    choice = input("\nSelect example (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(simple_query_example())
    elif choice == "2":
        asyncio.run(conversation_example())
    elif choice == "3":
        asyncio.run(session_management_example())
    else:
        print("Invalid choice")
