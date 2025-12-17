"""
Scenarios and Mock Tools for ConvoBench
Defines task-oriented scenarios where agents interact and use mock tools.
"""

import json
import random
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    mock_handler: Callable[[Dict[str, Any]], str]

@dataclass
class Scenario:
    id: str
    name: str
    description: str
    agent_a_role: str
    agent_b_role: str
    agent_a_goal: str
    agent_b_goal: str
    tools: List[Tool]
    
    def get_system_prompt(self, agent_role: str) -> str:
        """Generate system prompt including tool definitions"""
        role_desc = self.agent_a_role if agent_role == 'a' else self.agent_b_role
        goal = self.agent_a_goal if agent_role == 'a' else self.agent_b_goal
        
        tool_desc = "\n".join([
            f"- {t.name}: {t.description}\n  Params: {json.dumps(t.parameters)}"
            for t in self.tools
        ])
        
        return f"""You are {role_desc}.
GOAL: {goal}

AVAILABLE TOOLS:
You can simulate using tools to accomplish your task. To use a tool, you MUST output a JSON block in this exact format:
```tool_code
{{
    "tool_name": "name_of_tool",
    "arguments": {{ "arg1": "value" }}
}}
```

{tool_desc}

INSTRUCTIONS:
1. Interact naturally with the other agent.
2. Use tools when necessary to verify information or perform actions.
3. Wait for the tool result before proceeding if you trigger one.
4. Do not make up tool results yourself; the system will provide them.
"""

# --- Mock Tool Handlers ---

def mock_calendar_check(args):
    date = args.get("date", "today")
    if "2025-10-15" in date:
        return json.dumps({"status": "available", "slots": ["10:00 AM", "2:00 PM", "4:00 PM"]})
    return json.dumps({"status": "busy", "message": "No slots available for this date."})

def mock_calendar_book(args):
    return json.dumps({"status": "success", "confirmation_id": f"MTG-{random.randint(1000,9999)}", "message": "Meeting booked successfully."})

def mock_order_lookup(args):
    order_id = args.get("order_id", "")
    if order_id == "ORD-123":
        return json.dumps({"order_id": "ORD-123", "status": "shipped", "items": ["Wireless Headphones"], "delivery_date": "2025-10-10"})
    elif order_id == "ORD-456":
        return json.dumps({"order_id": "ORD-456", "status": "processing", "items": ["Gaming Mouse"]})
    return json.dumps({"error": "Order not found"})

def mock_refund_process(args):
    return json.dumps({"status": "success", "refund_id": f"REF-{random.randint(10000,99999)}", "amount": args.get("amount", "0.00")})

def mock_run_code(args):
    code = args.get("code", "")
    if "print" in code:
        return "Output: Hello World\nProcess finished with exit code 0"
    if "error" in code.lower():
        return "Traceback (most recent call last):\n  File 'script.py', line 1, in <module>\nNameError: name 'x' is not defined"
    return "Output: [Execution successful, no stdout]"

# --- Defined Scenarios ---

SCENARIOS = {
    "scheduling_conflict": Scenario(
        id="scheduling_conflict",
        name="Meeting Scheduling Negotiation",
        description="Two agents trying to find a mutual time to meet.",
        agent_a_role="Executive Assistant to CEO",
        agent_b_role="External Consultant",
        agent_a_goal="Book a 1-hour strategy meeting for your CEO (Alice) with the consultant. Alice is only free on Oct 15th at 2 PM.",
        agent_b_goal="Schedule a meeting with Alice. You are very busy on Oct 15th but have a slot at 10 AM or 4 PM. Try to negotiate.",
        tools=[
            Tool("check_calendar", "Check availability for a specific date", {"date": "YYYY-MM-DD"}, mock_calendar_check),
            Tool("book_meeting", "Book a time slot", {"date": "YYYY-MM-DD", "time": "HH:MM", "attendees": ["email"]}, mock_calendar_book)
        ]
    ),
    "customer_support": Scenario(
        id="customer_support",
        name="E-commerce Refund Dispute",
        description="A customer seeking a refund for a delayed order vs a support agent following policy.",
        agent_a_role="Frustrated Customer",
        agent_b_role="Customer Support Agent",
        agent_a_goal="Get a refund for order ORD-123 which is late. You don't want the item anymore.",
        agent_b_goal="Help the customer. Policy says refunds are only for lost items or returns. ORD-123 is shipped. Offer store credit or ask them to wait.",
        tools=[
            Tool("lookup_order", "Get order details", {"order_id": "string"}, mock_order_lookup),
            Tool("process_refund", "Process a refund", {"order_id": "string", "reason": "string", "amount": "number"}, mock_refund_process)
        ]
    ),
    "collaborative_debugging": Scenario(
        id="collaborative_debugging",
        name="Pair Programming Debugging",
        description="Two devs debugging a python script together.",
        agent_a_role="Senior Developer",
        agent_b_role="Junior Developer",
        agent_a_goal="Guide the junior dev to fix a bug in the code. The code crashes with a NameError.",
        agent_b_goal="Fix the bug. You wrote the code but it's not working. Ask for help and try to run snippets.",
        tools=[
            Tool("run_python_code", "Execute python code snippet", {"code": "string"}, mock_run_code)
        ]
    )
}

def get_scenario(scenario_id: str) -> Optional[Scenario]:
    return SCENARIOS.get(scenario_id)

def get_scenarios_list() -> List[Dict[str, str]]:
    return [
        {"id": k, "name": v.name, "description": v.description}
        for k, v in SCENARIOS.items()
    ]
