"""
Benchmark Evaluator for ConvoBench
Evaluates individual model performance in conversations to build a leaderboard.
"""

import json
import os
from typing import Dict, Any, List
from openai import OpenAI

class BenchmarkEvaluator:
    def __init__(self, api_key: str, base_url: str = "https://integrate.api.nvidia.com/v1"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = "moonshotai/kimi-k2-thinking"

    def evaluate_model_performance(self, conversation: Dict[str, Any], target_agent: str) -> Dict[str, Any]:
        """
        Evaluates a specific agent's performance in a conversation.
        target_agent: 'a' or 'b'
        """
        
        # Extract context
        messages = conversation.get("messages", [])
        seed_prompt = conversation.get("metadata", {}).get("seed_prompt", "")
        model_name = conversation["agents"][f"agent_{target_agent}"]["model"]
        
        # Format conversation for the evaluator
        transcript = ""
        target_role_name = f"Agent {target_agent.upper()}"
        
        for msg in messages:
            role_label = f"Agent {msg['role'].replace('agent_', '').upper()}"
            content = msg.get("content", "")
            transcript += f"[{role_label}]: {content}\n\n"

        prompt = f"""You are an expert judge evaluating the performance of an AI model in a conversation.

CONVERSATION CONTEXT:
The conversation started with the seed prompt: "{seed_prompt}"
You are evaluating the performance of **{target_role_name}** ({model_name}).

TRANSCRIPT:
---
{transcript}
---

EVALUATION CRITERIA:
1. **Instruction Following**: Did the model stay on topic and follow the implicit/explicit instructions of the conversation flow?
2. **Coherence**: Were the responses logical, consistent, and contextually appropriate?
3. **Reasoning**: Did the model demonstrate sound logic and depth of thought?
4. **Engagement**: Was the model's contribution interesting, promoting further dialogue?
5. **Overall Quality**: An aggregate score of the model's performance.

TASK:
Analyze {target_role_name}'s performance specifically. Ignore the other agent's quality unless it directly impacts {target_role_name}'s ability to respond.
If {target_role_name} outputted gibberish, repetition, or hallucinated heavily, penalize the score significantly.

OUTPUT FORMAT:
Return a JSON object with the following keys. Scores should be between 0.0 and 1.0.
{{
    "score_overall": 0.0,
    "score_coherence": 0.0,
    "score_reasoning": 0.0,
    "score_engagement": 0.0,
    "score_instruction_following": 0.0,
    "critique": "A brief explanation of the scores."
}}
"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4096,
                response_format={"type": "json_object"}
            )
            
            response_text = completion.choices[0].message.content
            # Cleanup markdown if present
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end]
                
            metrics = json.loads(response_text.strip())
            
            # Enrich with metadata
            metrics["conversation_id"] = conversation.get("id")
            metrics["model_name"] = model_name
            metrics["role"] = target_agent
            metrics["metrics_json"] = {"critique": metrics.get("critique", "")}
            metrics["timestamp"] = conversation.get("metadata", {}).get("end_time")
            
            return metrics

        except Exception as e:
            print(f"Error evaluating model {target_agent}: {e}")
            return {
                "score_overall": 0.0,
                "score_coherence": 0.0,
                "score_reasoning": 0.0,
                "score_engagement": 0.0,
                "score_instruction_following": 0.0,
                "metrics_json": {"error": str(e)},
                "conversation_id": conversation.get("id"),
                "model_name": model_name,
                "role": target_agent,
                "timestamp": conversation.get("metadata", {}).get("end_time")
            }
