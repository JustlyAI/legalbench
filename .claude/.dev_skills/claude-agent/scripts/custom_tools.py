#!/usr/bin/env python3
"""Template for creating custom tools and in-process MCP servers"""

import asyncio
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from claude_agent_sdk import (
    tool, create_sdk_mcp_server, ClaudeSDKClient, ClaudeAgentOptions,
    AssistantMessage, TextBlock, ResultMessage
)

# Simple tool with basic type mapping
@tool("calculate", "Perform mathematical calculations", {
    "expression": str,
    "precision": int
})
async def calculate(args: Dict[str, Any]) -> Dict[str, Any]:
    """Simple calculation tool"""
    try:
        precision = args.get("precision", 2)
        result = eval(args["expression"], {"__builtins__": {}})
        formatted = f"{result:.{precision}f}"
        return {
            "content": [{
                "type": "text",
                "text": f"Result: {args['expression']} = {formatted}"
            }]
        }
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Calculation error: {str(e)}"
            }],
            "is_error": True
        }

# Tool with Pydantic validation
class DatabaseQuery(BaseModel):
    """Schema for database query tool"""
    table: str = Field(description="Table name")
    query: str = Field(description="SQL query string")
    limit: int = Field(default=100, description="Result limit")

@tool("database_query", "Query database with validation", DatabaseQuery.model_json_schema())
async def database_query(args: Dict[str, Any]) -> Dict[str, Any]:
    """Database query tool with Pydantic validation"""
    try:
        # Validate inputs
        validated = DatabaseQuery(**args)
        
        # Mock database query (replace with actual DB logic)
        result = f"Querying {validated.table} with: {validated.query} (limit: {validated.limit})"
        
        return {
            "content": [{
                "type": "text",
                "text": result
            }]
        }
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Query error: {str(e)}"
            }],
            "is_error": True
        }

# Tool for file operations
@tool("process_files", "Batch process multiple files", {
    "pattern": str,
    "operation": str,
    "output_dir": str
})
async def process_files(args: Dict[str, Any]) -> Dict[str, Any]:
    """Batch file processing tool"""
    from pathlib import Path
    import glob
    
    pattern = args["pattern"]
    operation = args["operation"]
    output_dir = Path(args.get("output_dir", "./processed"))
    
    try:
        output_dir.mkdir(exist_ok=True)
        files = glob.glob(pattern)
        
        if not files:
            return {
                "content": [{
                    "type": "text",
                    "text": f"No files found matching pattern: {pattern}"
                }],
                "is_error": True
            }
        
        processed = []
        for file_path in files:
            # Mock processing (replace with actual logic)
            processed.append(Path(file_path).name)
        
        return {
            "content": [{
                "type": "text",
                "text": f"Processed {len(processed)} files: {', '.join(processed)}"
            }]
        }
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Processing error: {str(e)}"
            }],
            "is_error": True
        }

# Create in-process MCP server
def create_custom_server():
    """Create an SDK MCP server with custom tools"""
    return create_sdk_mcp_server(
        name="custom_tools",
        version="1.0.0",
        tools=[calculate, database_query, process_files]
    )

# Example usage with custom tools
async def demo_custom_tools():
    """Demonstrates using custom tools with Claude"""
    
    # Create the MCP server
    custom_server = create_custom_server()
    
    # Configure agent with custom tools
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        max_turns=5,
        # Register the MCP server
        mcp_servers={"custom": custom_server},
        # Allow specific tools (use mcp__{server}__{tool} naming)
        allowed_tools=[
            "mcp__custom__calculate",
            "mcp__custom__database_query",
            "mcp__custom__process_files",
            "Read"  # Also allow built-in Read tool
        ],
        system_prompt="You are an assistant with custom calculation and database tools."
    )
    
    async with ClaudeSDKClient(options=options) as client:
        # Query that will use custom tools
        await client.query("Calculate the compound interest on $10,000 at 5% for 10 years")
        
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)
            elif isinstance(message, ResultMessage):
                print(f"\n\nTotal cost: ${message.total_cost_usd:.4f}")
                break

# Hybrid approach: SDK tools + external MCP servers
def create_hybrid_config():
    """Example of combining SDK and external MCP servers"""
    
    # Create SDK server for trusted internal tools
    internal_server = create_sdk_mcp_server(
        name="internal",
        version="1.0.0",
        tools=[calculate]  # Fast, trusted tools
    )
    
    # Configure with both SDK and external servers
    return ClaudeAgentOptions(
        mcp_servers={
            "internal": internal_server,  # In-process SDK server
            "external": {  # External stdio server
                "type": "stdio",
                "command": "python",
                "args": ["-m", "external_mcp_server"]
            }
        },
        allowed_tools=[
            "mcp__internal__calculate",  # SDK tool
            "mcp__external__api_call"    # External tool
        ]
    )

if __name__ == "__main__":
    print("Custom Tool Creation Examples\n")
    print("Running demo with custom tools...")
    asyncio.run(demo_custom_tools())
