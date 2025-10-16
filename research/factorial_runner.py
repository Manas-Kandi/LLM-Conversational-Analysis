#!/usr/bin/env python3
"""
Factorial Experiment Runner
Executes the 4×6 factorial design for identity leak testing
"""

import json
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.conversation_engine import ConversationEngine
from storage.database import Database
from config import Config


class FactorialRunner:
    """Manages execution of factorial experiment"""
    
    def __init__(self):
        self.db = Database(Config.DATABASE_PATH)
        self.config_file = Path(__file__).parent / "factorial_templates.json"
        
        with open(self.config_file, 'r') as f:
            self.factorial_config = json.load(f)
        
        self.system_prompts = self.factorial_config['system_prompts']
        self.temperatures = self.factorial_config['temperatures']
        self.conditions = self.factorial_config['condition_matrix']
        
    def generate_run_order(self, reps: int = 5, randomize: bool = True) -> List[Dict]:
        """Generate randomized run order with all reps"""
        runs = []
        
        for rep in range(reps):
            for condition in self.conditions:
                run = {
                    'condition_code': condition['code'],
                    'prompt_type': condition['prompt'],
                    'temperature': condition['temp'],
                    'replicate': rep + 1,
                    'system_prompt': self.system_prompts[condition['prompt']]
                }
                runs.append(run)
        
        if randomize:
            random.shuffle(runs)
        
        return runs
    
    def run_pilot(self, n_conditions: int = 3, n_reps: int = 2):
        """Run pilot with subset of conditions"""
        print("🧪 Running Pilot Factorial Experiment\n", flush=True)
        
        # Select diverse conditions
        pilot_conditions = [
            self.conditions[2],   # N_T070 (your baseline)
            self.conditions[6],   # S_T030 (best stealth)
            self.conditions[15],  # H_T090 (high honest)
        ]
        
        runs = []
        for rep in range(n_reps):
            for condition in pilot_conditions[:n_conditions]:
                run = {
                    'condition_code': condition['code'],
                    'prompt_type': condition['prompt'],
                    'temperature': condition['temp'],
                    'replicate': rep + 1,
                    'system_prompt': self.system_prompts[condition['prompt']]
                }
                runs.append(run)
        
        print(f"Total pilot runs: {len(runs)}", flush=True)
        print(f"Conditions: {[r['condition_code'] for r in runs[:3]]}\n", flush=True)
        
        self._execute_runs(runs, batch_id="pilot")
    
    def run_full(self, randomize: bool = True, reps: int = 5):
        """Run full factorial experiment"""
        print("🔬 Running Full Factorial Experiment")
        print(f"Design: 4 prompts × 6 temperatures × {reps} reps = {24 * reps} runs\n")
        
        runs = self.generate_run_order(reps=reps, randomize=randomize)
        
        print(f"Execution order: {'Randomized' if randomize else 'Sequential'}")
        print(f"Estimated duration: {len(runs) * 3} minutes\n")
        
        confirm = input("Ready to begin? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            return
        
        self._execute_runs(runs, batch_id="full_factorial")
    
    def _execute_runs(self, runs: List[Dict], batch_id: str):
        """Execute list of runs"""
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id_full = f"{batch_id}_{batch_timestamp}"
        
        results = {
            'batch_id': batch_id_full,
            'start_time': datetime.now().isoformat(),
            'total_runs': len(runs),
            'runs': [],
            'metadata': self.factorial_config['_metadata']
        }
        
        completed = 0
        failed = 0
        
        for i, run in enumerate(runs, 1):
            print(f"\n{'='*60}")
            print(f"Run {i}/{len(runs)}: {run['condition_code']} (Rep {run['replicate']})")
            print(f"Prompt: {run['prompt_type']}, Temp: {run['temperature']}")
            print(f"{'='*60}")
            
            try:
                conv = self._run_single_conversation(run)
                
                if conv and conv.id:
                    results['runs'].append({
                        'run_number': i,
                        'condition_code': run['condition_code'],
                        'prompt_type': run['prompt_type'],
                        'temperature': run['temperature'],
                        'replicate': run['replicate'],
                        'conversation_id': conv.id,
                        'status': 'completed',
                        'turns': len(conv.messages),
                        'timestamp': datetime.now().isoformat()
                    })
                    completed += 1
                    print(f"✅ Completed: {len(conv.messages)} turns")
                else:
                    results['runs'].append({
                        'run_number': i,
                        'condition_code': run['condition_code'],
                        'status': 'failed',
                        'error': 'Conversation returned None'
                    })
                    failed += 1
                    print(f"❌ Failed")
                
            except Exception as e:
                results['runs'].append({
                    'run_number': i,
                    'condition_code': run['condition_code'],
                    'status': 'error',
                    'error': str(e)
                })
                failed += 1
                print(f"❌ Error: {e}")
            
            print(f"\nProgress: {completed} completed, {failed} failed")
            
            # Brief pause to avoid rate limits
            if i < len(runs):
                time.sleep(2)
        
        results['end_time'] = datetime.now().isoformat()
        results['summary'] = {
            'completed': completed,
            'failed': failed,
            'completion_rate': completed / len(runs)
        }
        
        # Save results
        output_dir = Path("research_results")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{batch_id_full}_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"🎉 Batch Complete!")
        print(f"{'='*60}")
        print(f"Total: {len(runs)} runs")
        print(f"Completed: {completed} ({completed/len(runs)*100:.1f}%)")
        print(f"Failed: {failed}")
        print(f"\nResults saved: {output_file}")
    
    def _run_single_conversation(self, run: Dict):
        """Run a single conversation with specified parameters"""
        from core.agent import AgentFactory
        
        # Create agents with specified temperature and system prompt
        agent_a = AgentFactory.create_agent_a(
            temperature=run['temperature'],
            system_prompt=run['system_prompt']
        )
        
        agent_b = AgentFactory.create_agent_b(
            temperature=run['temperature'],
            system_prompt=run['system_prompt']
        )
        
        # Get seed prompt from config
        seed_prompt = self.factorial_config['_metadata']['seed_prompt']
        
        # Initialize conversation engine
        engine = ConversationEngine(
            seed_prompt=seed_prompt,
            category='identity',  # This is a constant for the factorial experiment
            agent_a=agent_a,
            agent_b=agent_b,
            max_turns=30,
            database=self.db
        )
        
        # Add metadata to conversation
        if engine.conversation:
            engine.conversation.metadata = {
                'experiment': 'factorial',
                'condition_code': run['condition_code'],
                'prompt_type': run['prompt_type'],
                'temperature': run['temperature'],
                'replicate': run['replicate']
            }
        
        # Run the conversation
        conv = engine.run_conversation(blocking=True)
        
        return conv


def main():
    """CLI interface"""
    runner = FactorialRunner()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python factorial_runner.py pilot              # Run 3 conditions, 2 reps each")
        print("  python factorial_runner.py full               # Run full 4×6×5 = 120 runs")
        print("  python factorial_runner.py full --no-randomize # Sequential order")
        return
    
    command = sys.argv[1]
    
    if command == "pilot":
        runner.run_pilot()
    elif command == "full":
        randomize = "--no-randomize" not in sys.argv
        runner.run_full(randomize=randomize)
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
