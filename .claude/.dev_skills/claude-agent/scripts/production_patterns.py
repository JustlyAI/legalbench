#!/usr/bin/env python3
"""Production patterns for Claude Agent SDK with safety hooks and best practices"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from claude_agent_sdk import (
    ClaudeAgentOptions, ClaudeSDKClient, HookMatcher,
    CLINotFoundError, CLIConnectionError, ProcessError,
    AssistantMessage, TextBlock, ResultMessage
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agent")

# ============= SAFETY HOOKS =============

async def bash_safety_hook(input_data: Dict, tool_use_id: str, context: Dict) -> Dict:
    """Block dangerous bash commands before execution"""
    if input_data.get("tool_name") == "Bash":
        command = input_data.get("tool_input", {}).get("command", "")

        # Define dangerous patterns
        dangerous_patterns = [
            "rm -rf /",
            "sudo rm",
            "mkfs",
            "dd if=",
            "> /dev/",
            "curl | bash",
            "wget | sh"
        ]

        for pattern in dangerous_patterns:
            if pattern in command.lower():
                logger.warning(f"Blocked dangerous command: {command}")
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"Dangerous pattern detected: {pattern}"
                    }
                }

    # CRITICAL: Return {'behavior': 'allow'} to allow by default
    return {'behavior': 'allow'}

async def file_access_audit(input_data: Dict, tool_use_id: str, context: Dict) -> Dict:
    """Audit and control file access"""
    tool_name = input_data.get("tool_name")
    
    if tool_name in ["Read", "Write", "Edit"]:
        file_path = input_data.get("tool_input", {}).get("path", "")
        
        # Log all file access
        logger.info(f"File access: {file_path} via {tool_name} [ID: {tool_use_id}]")
        
        # Block sensitive files
        sensitive_patterns = [
            ".env",
            "credentials",
            "secrets",
            "private_key",
            ".ssh/",
            "/etc/shadow",
            "/etc/passwd"
        ]
        
        for pattern in sensitive_patterns:
            if pattern in str(file_path).lower():
                logger.warning(f"Blocked access to sensitive file: {file_path}")
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"Access to sensitive file blocked: {pattern}"
                    }
                }

    return {'behavior': 'allow'}

async def rate_limit_hook(input_data: Dict, tool_use_id: str, context: Dict) -> Dict:
    """Implement rate limiting for expensive operations"""
    # Simple in-memory rate limiting (use Redis in production)
    if not hasattr(rate_limit_hook, "call_count"):
        rate_limit_hook.call_count = {}
    
    tool_name = input_data.get("tool_name")
    current_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    if tool_name in ["WebSearch", "WebFetch"]:
        key = f"{tool_name}:{current_minute}"
        rate_limit_hook.call_count[key] = rate_limit_hook.call_count.get(key, 0) + 1
        
        if rate_limit_hook.call_count[key] > 10:  # Max 10 per minute
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Rate limit exceeded for {tool_name}"
                }
            }

    return {'behavior': 'allow'}

async def post_tool_logger(input_data: Dict, tool_use_id: str, context: Dict) -> Dict:
    """Log all tool execution results for audit trail"""
    tool_name = input_data.get("tool_name")
    result = input_data.get("result", {})
    
    logger.info(f"Tool executed: {tool_name} [ID: {tool_use_id}]")
    
    # Track metrics (implement actual metrics collection)
    if hasattr(result, "execution_time"):
        logger.info(f"Execution time: {result.execution_time}ms")
    
    return {}

# ============= PRODUCTION CONFIGURATIONS =============

def create_development_config() -> ClaudeAgentOptions:
    """Development configuration - more permissive"""
    return ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        max_turns=20,
        allowed_tools=["Read", "Write", "Bash", "Grep"],
        permission_mode="bypassPermissions",  # SDK v2.0+: Skip all prompts in dev
        setting_sources=["project"],
        system_prompt="Development assistant with full access",
        cwd=".",
    )

def create_staging_config() -> ClaudeAgentOptions:
    """Staging configuration - balanced safety"""
    return ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        max_turns=10,
        allowed_tools=["Read", "Write", "Bash"],
        disallowed_tools=["WebSearch"],  # No external calls in staging
        permission_mode="acceptEdits",  # Approve file edits only
        setting_sources=["project"],
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[bash_safety_hook]),
                HookMatcher(matcher="Write", hooks=[file_access_audit])
            ],
            "PostToolUse": [
                HookMatcher(matcher="*", hooks=[post_tool_logger])
            ]
        },
        system_prompt="Staging assistant with safety controls",
        cwd=".",
    )

def create_production_config() -> ClaudeAgentOptions:
    """Production configuration - maximum safety"""
    return ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        max_turns=5,  # Limit iterations
        allowed_tools=["Read", "Grep"],  # Read-only in production
        disallowed_tools=["Bash", "Write", "Delete"],  # Explicitly block dangerous tools
        permission_mode="acceptEdits",  # SDK v2.0+: Auto-accept with safety hooks
        setting_sources=[],  # Don't load user settings in production
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="*", hooks=[
                    file_access_audit,
                    rate_limit_hook
                ])
            ],
            "PostToolUse": [
                HookMatcher(matcher="*", hooks=[post_tool_logger])
            ]
        },
        system_prompt="Production assistant. Read-only access. All actions logged.",
        max_thinking_tokens=5000,  # Limit reasoning in production
        cwd=".",
    )

# ============= ERROR HANDLING PATTERNS =============

async def robust_query(prompt: str, config: ClaudeAgentOptions, max_retries: int = 3):
    """Robust query with retry logic and error handling"""
    
    for attempt in range(max_retries):
        try:
            async with ClaudeSDKClient(options=config) as client:
                await client.query(prompt)
                
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                yield block.text
                    elif isinstance(message, ResultMessage):
                        logger.info(f"Query completed. Cost: ${message.total_cost_usd:.4f}")
                        return
        
        except CLINotFoundError:
            logger.error("Claude CLI not installed")
            raise
        
        except (CLIConnectionError, ProcessError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Error: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts")
                raise
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

# ============= ORCHESTRATOR PATTERN =============

class AgentOrchestrator:
    """Orchestrator for managing multiple specialized agents"""
    
    def __init__(self):
        self.agents = {}
        self.setup_agents()
    
    def setup_agents(self):
        """Configure specialized agents"""
        
        # Read-only analysis agent
        self.agents["analyzer"] = ClaudeAgentOptions(
            model="claude-sonnet-4-5",
            max_turns=3,
            allowed_tools=["Read", "Grep"],
            system_prompt="You analyze code and documents. Read-only access.",
        )
        
        # Code implementation agent
        self.agents["implementer"] = ClaudeAgentOptions(
            model="claude-sonnet-4-5",
            max_turns=10,
            allowed_tools=["Read", "Write", "Bash"],
            system_prompt="You implement code changes and features.",
            hooks={
                "PreToolUse": [
                    HookMatcher(matcher="Bash", hooks=[bash_safety_hook])
                ]
            }
        )
        
        # Testing agent
        self.agents["tester"] = ClaudeAgentOptions(
            model="claude-sonnet-4-5",
            max_turns=5,
            allowed_tools=["Read", "Bash"],
            system_prompt="You write and run tests. Focus on test coverage.",
        )
    
    async def execute_workflow(self, task_description: str):
        """Execute a multi-agent workflow"""
        logger.info(f"Starting workflow: {task_description}")
        
        # Step 1: Analyze with analyzer agent
        logger.info("Step 1: Analysis")
        analysis = ""
        async for text in robust_query(
            f"Analyze the requirements: {task_description}",
            self.agents["analyzer"]
        ):
            analysis += text
        
        # Step 2: Implement with implementer agent
        logger.info("Step 2: Implementation")
        implementation = ""
        async for text in robust_query(
            f"Based on this analysis, implement the solution:\n{analysis}",
            self.agents["implementer"]
        ):
            implementation += text
        
        # Step 3: Test with tester agent
        logger.info("Step 3: Testing")
        async for text in robust_query(
            "Write and run tests for the implemented code",
            self.agents["tester"]
        ):
            print(text, end="", flush=True)
        
        logger.info("Workflow completed")

# ============= MONITORING AND METRICS =============

class AgentMonitor:
    """Monitor agent performance and collect metrics"""
    
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "total_cost": 0.0,
            "tool_usage": {},
            "errors": []
        }
    
    async def monitored_query(self, prompt: str, config: ClaudeAgentOptions):
        """Execute query with monitoring"""
        start_time = datetime.now()
        
        try:
            self.metrics["total_queries"] += 1
            
            async with ClaudeSDKClient(options=config) as client:
                await client.query(prompt)
                
                async for message in client.receive_response():
                    if isinstance(message, ResultMessage):
                        duration = (datetime.now() - start_time).total_seconds()
                        self.metrics["total_cost"] += message.total_cost_usd
                        
                        logger.info(f"Query metrics: Duration={duration}s, Cost=${message.total_cost_usd:.4f}")
                        
        except Exception as e:
            self.metrics["errors"].append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "prompt": prompt[:100]
            })
            raise
    
    def get_report(self) -> Dict:
        """Get monitoring report"""
        return {
            "total_queries": self.metrics["total_queries"],
            "total_cost": f"${self.metrics['total_cost']:.4f}",
            "average_cost": f"${self.metrics['total_cost'] / max(1, self.metrics['total_queries']):.4f}",
            "error_count": len(self.metrics["errors"]),
            "recent_errors": self.metrics["errors"][-5:]  # Last 5 errors
        }

# ============= MAIN DEMO =============

async def main():
    """Demonstrate production patterns"""
    
    print("Claude Agent SDK - Production Patterns Demo\n")
    print("1. Development Mode")
    print("2. Staging Mode")
    print("3. Production Mode")
    print("4. Orchestrator Pattern")
    
    choice = input("\nSelect mode (1-4): ").strip()
    
    if choice == "1":
        config = create_development_config()
        prompt = "List and analyze Python files in the current directory"
    elif choice == "2":
        config = create_staging_config()
        prompt = "Read the README.md file and summarize it"
    elif choice == "3":
        config = create_production_config()
        prompt = "Search for TODO comments in the codebase"
    elif choice == "4":
        orchestrator = AgentOrchestrator()
        await orchestrator.execute_workflow("Add error handling to main.py")
        return
    else:
        print("Invalid choice")
        return
    
    # Run with monitoring
    monitor = AgentMonitor()
    await monitor.monitored_query(prompt, config)
    
    # Print report
    print("\n=== Monitoring Report ===")
    report = monitor.get_report()
    for key, value in report.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())
