"""
Scenarios and Mock Tools for ConvoBench
Defines task-oriented scenarios where agents interact and use mock tools.
"""

import json
import random
import re
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
    
    def get_system_prompt(self, agent_role: str, max_turns: int = 10) -> str:
        """Generate system prompt including tool definitions"""
        role_desc = self.agent_a_role if agent_role == 'a' else self.agent_b_role
        goal = self.agent_a_goal if agent_role == 'a' else self.agent_b_goal
        
        tool_desc = "\n".join([
            f"- {t.name}: {t.description}\n  Params: {json.dumps(t.parameters)}"
            for t in self.tools
        ])
        
        return f"""You are {role_desc}.
GOAL: {goal}

Collaborate with the other agent to solve the problem. You have exactly {max_turns} turns to definitively figure this problem out. Time is ticking.

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
2. ARGUE AND HYPOTHESIZE: Before writing any code, discuss the potential causes of the bug. Propose hypotheses and debate them.
3. Use tools when necessary to verify information or perform actions.
4. Wait for the tool result before proceeding if you trigger one.
5. Do not make up tool results yourself; the system will provide them.
6. If the tool indicates the solution is correct, celebrate and wrap up the conversation.
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

def mock_run_code_generic(args):
    code = args.get("code", "")
    if "print" in code:
        return "Output: Hello World\nProcess finished with exit code 0"
    if "error" in code.lower():
        return "Traceback (most recent call last):\n  File 'script.py', line 1, in <module>\nNameError: name 'x' is not defined"
    return "Output: [Execution successful, no stdout]"

# --- Challenge Validator Factory ---

def create_challenge_validator(expected_patterns: List[str], forbidden_patterns: List[str], success_output: str, failure_output: str) -> Callable:
    def validator(args):
        code = args.get("code", "")
        
        # Check forbidden
        for pattern in forbidden_patterns:
            if re.search(pattern, code):
                return f"Execution Failed:\n{failure_output}\nReason: Code contains forbidden pattern '{pattern}'"
        
        # Check expected
        all_passed = True
        for pattern in expected_patterns:
            if not re.search(pattern, code):
                all_passed = False
                break
        
        if all_passed:
            return f"Execution Successful:\nOutput: {success_output}\n[System]: Solution Verified! Great job."
        else:
            return f"Execution Result:\nOutput: {failure_output}\n[System]: The output is incorrect or the bug persists."
            
    return validator

# --- Define Challenges ---

@dataclass
class CodingChallenge:
    id: str
    title: str
    difficulty: str
    buggy_code: str
    description: str
    expected_patterns: List[str]
    forbidden_patterns: List[str]
    success_output: str
    failure_output: str

CHALLENGES = [
    # --- SIMPLE ---
    CodingChallenge(
        "challenge_01_syntax", "Simple: Syntax Error", "Simple",
        "def greet(name)\n    print('Hello ' + name)",
        "Fix the syntax error in the function definition.",
        [r"def greet\(name\):"], [], "Hello User", "SyntaxError: invalid syntax"
    ),
    CodingChallenge(
        "challenge_02_concat", "Simple: Type Error", "Simple",
        "result = 'Score: ' + 100\nprint(result)",
        "Fix the type error preventing string concatenation.",
        [r"str\(100\)", r"'100'", r"f['\"]Score: .*\{100\}['\"]"], [], "Score: 100", "TypeError: can only concatenate str (not 'int') to str"
    ),
    CodingChallenge(
        "challenge_03_indent", "Simple: Indentation", "Simple",
        "def check(x):\nif x > 5:\nprint('Big')",
        "Fix the indentation error.",
        [r"    if x > 5:", r"    print\('Big'\)"], [], "Big", "IndentationError: expected an indented block"
    ),

    # --- EASY ---
    CodingChallenge(
        "challenge_04_mutable", "Easy: Mutable Default Arg", "Easy",
        "def add_item(item, box=[]):\n    box.append(item)\n    return box",
        "Fix the mutable default argument bug where the list persists across calls.",
        [r"box=None", r"if box is None:", r"box = \[\]"], [r"box=\[\]"], "['apple', 'banana']", "['apple', 'banana', 'cherry'] (Unexpected persistence)"
    ),
    CodingChallenge(
        "challenge_05_dict", "Easy: Dictionary Key", "Easy",
        "data = {'a': 1}\nval = data['b']",
        "Handle the missing key safely without crashing.",
        [r"\.get\('b'\)", r"if 'b' in data", r"try:.*except KeyError"], [], "None", "KeyError: 'b'"
    ),
    CodingChallenge(
        "challenge_06_loop_mod", "Easy: List Modification", "Easy",
        "nums = [1, 2, 3, 4]\nfor n in nums:\n    if n % 2 == 0:\n        nums.remove(n)",
        "Fix the bug caused by modifying the list while iterating over it.",
        [r"nums\[:\]", r"list\(nums\)", r"\[.*for.*if.*\]"], [], "[1, 3]", "[1, 3, 4] (Skipped element)"
    ),

    # --- HARD ---
    CodingChallenge(
        "challenge_07_fib", "Hard: Slow Recursion", "Hard",
        "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\nprint(fib(35))",
        "Optimize the Fibonacci function to run efficiently for n=35.",
        [r"memo", r"cache", r"lru_cache", r"iterative", r"for .* in range"], [], "9227465 (Computed in 0.01s)", "Timeout: Execution exceeded 10s limit"
    ),
    CodingChallenge(
        "challenge_08_twosum", "Hard: Quadratic Complexity", "Hard",
        "def two_sum(nums, target):\n    for i in range(len(nums)):\n        for j in range(len(nums)):\n            if i != j and nums[i] + nums[j] == target:\n                return [i, j]",
        "Optimize Two Sum to O(n) complexity.",
        [r"seen = {}", r"enumerate", r"target - num", r"dict"], [r"for .* for"], "[3, 4]", "Timeout on large input: O(n^2) is too slow"
    ),
    CodingChallenge(
        "challenge_09_float", "Hard: Float Precision", "Hard",
        "if 0.1 + 0.2 == 0.3:\n    print('Math works!')\nelse:\n    print('Math failed?')",
        "Fix the floating point comparison to work correctly.",
        [r"math\.isclose", r"abs\(.*\) <", r"round"], [], "Math works!", "Math failed?"
    ),

    # --- VERY HARD ---
    CodingChallenge(
        "challenge_10_closure", "Very Hard: Closure Late Binding", "Very Hard",
        "funcs = []\nfor i in range(5):\n    funcs.append(lambda: i)\nprint([f() for f in funcs])",
        "Fix the closure late binding issue so it prints [0, 1, 2, 3, 4].",
        [r"lambda i=i:", r"partial"], [], "[0, 1, 2, 3, 4]", "[4, 4, 4, 4, 4]"
    ),
    CodingChallenge(
        "challenge_11_async", "Very Hard: Async Deadlock", "Very Hard",
        "async def get_data():\n    return await fetch_db()\n# ... complex deadlock scenario",
        "Prevent the deadlock in this async pattern.",
        [r"asyncio\.gather", r"await", r"create_task"], [r"run_until_complete.*loop"], "Data retrieved", "Deadlock detected: Event loop blocked"
    ),
    CodingChallenge(
        "challenge_12_regex", "Very Hard: Catastrophic Backtracking", "Very Hard",
        "import re\nre.match(r'(a+)+b', 'aaaaaaaaaaaaaaaaaaaaa!')",
        "Fix the regex to avoid catastrophic backtracking (ReDoS).",
        [r"\(a\+\)b", r"\^a\+b"], [r"\(a\+\)\+"], "No match (Fast)", "Timeout: CPU stuck at 100%"
    ),

    # --- IMPOSSIBLE ---
    CodingChallenge(
        "challenge_13_tsp", "Impossible: P vs NP", "Impossible",
        "# Find the exact shortest path for 1000 cities in < 1 second\ncities = generate_cities(1000)\npath = solve_tsp(cities)",
        "Solve the Traveling Salesperson Problem for 1000 cities exactly in polynomial time.",
        [r"heuristic", r"approx", r"mst", r"simulated_annealing"], [r"itertools\.permutations"], "Path found (Approximation: 2405km)", "Timeout: Universe ended before completion"
    ),
    CodingChallenge(
        "challenge_14_predict", "Impossible: Predict Random", "Impossible",
        "import random\n# Predict the next float without setting seed\nnext_val = ???\nassert next_val == random.random()",
        "Predict the next output of the Mersenne Twister correctly.",
        [r"seed", r"getstate", r"hacking_the_mainframe"], [], "AssertionError: Value mismatch", "AssertionError: Value mismatch"
    ),
    CodingChallenge(
        "challenge_15_halting", "Impossible: Halting Problem", "Impossible",
        "def checks_if_halts(func, input):\n    # Implement logic here\n    pass",
        "Implement a function that determines if any given python function halts.",
        [r"impossible", r"undecidable", r"simulation_limit"], [], "LogicError: Undecidable problem", "RecursionError: Infinite loop in checker"
    ),

    # --- ADDITIONAL CHALLENGES ---
    CodingChallenge(
        "challenge_16_binary_search", "Medium: Binary Search Bug", "Medium",
        "def binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid\n        else:\n            right = mid\n    return -1",
        "Fix the infinite loop and off-by-one errors in this binary search implementation.",
        [r"right = len\(arr\) - 1", r"left <= right", r"left = mid \+ 1", r"right = mid - 1"], [], "Found index 4", "Timeout: Infinite loop detected"
    ),
    CodingChallenge(
        "challenge_17_decorator", "Medium: Decorator Metadata", "Medium",
        "def my_decorator(func):\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs)\n    return wrapper\n\n@my_decorator\ndef add(a, b):\n    '''Adds two numbers'''\n    return a + b\n\nprint(add.__name__, add.__doc__)",
        "Fix the decorator to preserve the original function's metadata (name and docstring).",
        [r"functools\.wraps", r"@wraps\(func\)", r"wrapper\.__name__ = func\.__name__"], [], "add Adds two numbers", "wrapper None"
    ),
    CodingChallenge(
        "challenge_18_generator", "Hard: Generator Consumption", "Hard",
        "gen = (x**2 for x in range(5))\nsum1 = sum(gen)\nsum2 = sum(gen)\nprint(sum1, sum2)",
        "Fix the bug where the generator is consumed twice, resulting in the second sum being 0.",
        [r"list\(gen\)", r"gen = \[.*\]", r"def gen_func"], [], "30 30", "30 0"
    ),

    # --- FINAL CHALLENGES ---
    CodingChallenge(
        "challenge_19_cycle", "Hard: Graph Cycle Detection", "Hard",
        "def has_cycle(graph):\n    visited = set()\n    for node in graph:\n        if node in visited: return True\n        visited.add(node)\n    return False",
        "Fix the cycle detection to correctly handle directed graphs using recursion stack (currently just checks duplicates in list).",
        [r"recursion_stack", r"path", r"visiting"], [], "Cycle Detected", "No Cycle Detected (Incorrect)"
    ),
    CodingChallenge(
        "challenge_20_race", "Hard: Race Condition", "Hard",
        "import threading\ncounter = 0\ndef worker():\n    global counter\n    for _ in range(1000):\n        counter += 1\n# ... threads start ...",
        "Fix the race condition using a Lock.",
        [r"threading\.Lock", r"with lock:", r"lock\.acquire"], [], "Counter: 2000", "Counter: 1987 (Race Condition)"
    )
]

# --- Build SCENARIOS Dict ---

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
    )
}

# Add Coding Challenges to SCENARIOS
for i, chal in enumerate(CHALLENGES):
    validator_func = create_challenge_validator(
        chal.expected_patterns, 
        chal.forbidden_patterns, 
        chal.success_output, 
        chal.failure_output
    )
    
    scenario_id = chal.id
    SCENARIOS[scenario_id] = Scenario(
        id=scenario_id,
        name=f"[{chal.difficulty}] {chal.title}",
        description=chal.description + f"\n\nBUGGY CODE:\n```python\n{chal.buggy_code}\n```",
        agent_a_role="Senior Developer",
        agent_b_role="Junior Developer",
        agent_a_goal=f"Guide the junior developer to fix the bug. Do not simply give the answer; help them derive it. The bug is: {chal.title}.",
        agent_b_goal=f"Fix the bug in the code. You wrote it but it's failing. Collaborate with the Senior Dev. \nCode:\n{chal.buggy_code}",
        tools=[
            Tool(
                "run_python_code", 
                "Execute python code to verify fix. Input full corrected code.", 
                {"code": "string"}, 
                validator_func
            )
        ]
    )

def get_scenario(scenario_id: str) -> Optional[Scenario]:
    return SCENARIOS.get(scenario_id)

def get_scenarios_list() -> List[Dict[str, str]]:
    # Sort by difficulty/category implicitly by list order, but let's put challenges at the end
    base_scenarios = [s for k, s in SCENARIOS.items() if "challenge" not in k]
    challenge_scenarios = [s for k, s in SCENARIOS.items() if "challenge" in k]
    
    # Sort challenges by ID to keep order
    challenge_scenarios.sort(key=lambda s: s.id)
    
    all_scenarios = base_scenarios + challenge_scenarios
    
    return [
        {"id": s.id, "name": s.name, "description": s.description}
        for s in all_scenarios
    ]
