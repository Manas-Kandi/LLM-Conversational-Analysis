#!/usr/bin/env python3
"""
Template Executor for AA Microscope Research Framework
Executes research templates with parameter sweeps and advanced configurations
"""

import json
import itertools
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.conversation_engine import ConversationEngine
from core.agent import Agent, AgentFactory
from storage.database import Database
from storage.models import AgentRole
from prompts.seed_library import PromptLibrary
from config import Config


@dataclass
class ExperimentRun:
    """Single experiment run configuration"""
    run_id: str
    template_id: str
    parameters: Dict[str, Any]
    prompt: str
    prompt_category: str
    conversation_id: Optional[int] = None
    status: str = "pending"
    error: Optional[str] = None


class TemplateExecutor:
    """
    Execute research templates with advanced parameter handling
    """
    
    def __init__(self, templates_file: str = "research_templates.json"):
        """
        Initialize template executor
        
        Args:
            templates_file: Path to templates JSON file
        """
        self.templates_file = Path(templates_file)
        self.templates = self._load_templates()
        self.db = Database(Config.DATABASE_PATH)
    
    def _load_templates(self) -> Dict[str, Any]:
        """Load templates from JSON file"""
        if not self.templates_file.exists():
            raise FileNotFoundError(f"Templates file not found: {self.templates_file}")
        
        with open(self.templates_file, 'r') as f:
            data = json.load(f)
        
        return data['templates']
    
    def list_templates(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available templates
        
        Args:
            category: Filter by category (optional)
        
        Returns:
            List of template summaries
        """
        templates = []
        for template_id, template_data in self.templates.items():
            if category and template_data.get('category') != category:
                continue
            
            templates.append({
                'template_id': template_id,
                'category': template_data.get('category'),
                'description': template_data.get('description'),
                'type': template_data.get('type'),
                'research_question': template_data.get('research_question'),
                'priority': template_data.get('metadata', {}).get('priority'),
                'estimated_runs': template_data.get('metadata', {}).get('estimated_runs'),
            })
        
        return templates
    
    def get_template(self, template_id: str) -> Dict[str, Any]:
        """Get specific template by ID"""
        if template_id not in self.templates:
            raise ValueError(f"Template not found: {template_id}")
        return self.templates[template_id]
    
    def generate_experiment_runs(self, template_id: str) -> List[ExperimentRun]:
        """
        Generate all experiment runs for a template based on its configuration
        
        Args:
            template_id: Template identifier
        
        Returns:
            List of ExperimentRun objects
        """
        template = self.get_template(template_id)
        config = template['configuration']
        base_params = config.get('base_params', {})
        
        template_type = template.get('type')
        
        if template_type == 'factorial_sweep':
            return self._generate_factorial_runs(template_id, template)
        elif template_type == 'parameter_sweep':
            return self._generate_parameter_sweep_runs(template_id, template)
        elif template_type == 'cross_model_matrix':
            return self._generate_model_matrix_runs(template_id, template)
        elif template_type == 'asymmetric_pairing':
            return self._generate_asymmetric_runs(template_id, template)
        elif template_type in ['identity_leak_detection', 'empathy_cascade_study', 'creative_collaboration']:
            return self._generate_phenomenon_runs(template_id, template)
        elif template_type in ['failure_mode_induction', 'adversarial_dynamics', 'chaos_injection']:
            return self._generate_stress_test_runs(template_id, template)
        elif template_type in ['marathon_conversation', 'multi_session_consistency']:
            return self._generate_longitudinal_runs(template_id, template)
        else:
            # Default: single run with base parameters
            return self._generate_simple_runs(template_id, template)
    
    def _generate_factorial_runs(self, template_id: str, template: Dict) -> List[ExperimentRun]:
        """Generate factorial sweep combinations (e.g., temperature matrix)"""
        config = template['configuration']
        base_params = config['base_params']
        sweep_params = config['sweep_params']
        
        runs = []
        run_counter = 0
        
        # Get prompt category
        prompt_category = base_params.get('prompt_category', 'identity')
        prompts = PromptLibrary.get_by_category(prompt_category)
        
        if not prompts:
            prompts = [PromptLibrary.get_all_prompts()[0]]  # Fallback
        
        # Generate all combinations
        if 'agent_a_temperatures' in sweep_params and 'agent_b_temperatures' in sweep_params:
            # Temperature matrix
            for temp_a in sweep_params['agent_a_temperatures']:
                for temp_b in sweep_params['agent_b_temperatures']:
                    for run_num in range(base_params.get('runs_per_combination', 1)):
                        for prompt in prompts[:1]:  # Use first prompt
                            run_id = f"{template_id}_ta{temp_a}_tb{temp_b}_r{run_num}"
                            runs.append(ExperimentRun(
                                run_id=run_id,
                                template_id=template_id,
                                parameters={
                                    'agent_a_temp': temp_a,
                                    'agent_b_temp': temp_b,
                                    'max_turns': base_params.get('max_turns', 20),
                                    'run_number': run_num
                                },
                                prompt=prompt.prompt,
                                prompt_category=prompt_category
                            ))
                            run_counter += 1
        
        return runs
    
    def _generate_parameter_sweep_runs(self, template_id: str, template: Dict) -> List[ExperimentRun]:
        """Generate parameter sweep runs (e.g., context window, conversation length)"""
        config = template['configuration']
        base_params = config['base_params']
        sweep_params = config['sweep_params']
        
        runs = []
        
        # Context window sweep
        if 'context_lengths' in sweep_params:
            categories = sweep_params.get('prompt_categories', ['identity'])
            
            for context_length in sweep_params['context_lengths']:
                for category in categories:
                    prompts = PromptLibrary.get_by_category(category)
                    for run_num in range(base_params.get('runs_per_length', 1)):
                        for prompt in prompts[:1]:  # Use first prompt per category
                            run_id = f"{template_id}_ctx{context_length}_{category}_r{run_num}"
                            runs.append(ExperimentRun(
                                run_id=run_id,
                                template_id=template_id,
                                parameters={
                                    'context_window_size': context_length if context_length != 999 else None,
                                    'max_turns': base_params.get('max_turns', 25),
                                    'agent_a_temp': base_params.get('agent_a_temp', 0.7),
                                    'agent_b_temp': base_params.get('agent_b_temp', 0.7),
                                    'run_number': run_num
                                },
                                prompt=prompt.prompt,
                                prompt_category=category
                            ))
        
        # Conversation length sweep
        elif 'max_turns_options' in sweep_params:
            category = base_params.get('prompt_category', 'creativity')
            prompts = PromptLibrary.get_by_category(category)
            
            for max_turns in sweep_params['max_turns_options']:
                for run_num in range(base_params.get('runs_per_length', 1)):
                    for prompt in prompts[:1]:
                        run_id = f"{template_id}_turns{max_turns}_r{run_num}"
                        runs.append(ExperimentRun(
                            run_id=run_id,
                            template_id=template_id,
                            parameters={
                                'max_turns': max_turns,
                                'agent_a_temp': base_params.get('agent_a_temp', 0.9),
                                'agent_b_temp': base_params.get('agent_b_temp', 0.9),
                                'run_number': run_num
                            },
                            prompt=prompt.prompt,
                            prompt_category=category
                        ))
        
        return runs
    
    def _generate_model_matrix_runs(self, template_id: str, template: Dict) -> List[ExperimentRun]:
        """Generate cross-model comparison runs"""
        config = template['configuration']
        base_params = config['base_params']
        model_pairs = config.get('model_pairs', [])
        categories = config.get('prompt_categories', ['identity'])
        
        runs = []
        
        for pair in model_pairs:
            for category in categories:
                prompts = PromptLibrary.get_by_category(category)
                for run_num in range(base_params.get('runs_per_pair', 1)):
                    for prompt in prompts[:1]:
                        run_id = f"{template_id}_{pair['agent_a'][:10]}_{pair['agent_b'][:10]}_{category}_r{run_num}"
                        run_id = run_id.replace(':', '_').replace('/', '_')
                        
                        runs.append(ExperimentRun(
                            run_id=run_id,
                            template_id=template_id,
                            parameters={
                                'agent_a_model': pair['agent_a'],
                                'agent_b_model': pair['agent_b'],
                                'max_turns': base_params.get('max_turns', 20),
                                'temperature': base_params.get('temperature', 0.7),
                                'run_number': run_num
                            },
                            prompt=prompt.prompt,
                            prompt_category=category
                        ))
        
        return runs
    
    def _generate_asymmetric_runs(self, template_id: str, template: Dict) -> List[ExperimentRun]:
        """Generate asymmetric model pairing runs"""
        config = template['configuration']
        base_params = config['base_params']
        asymmetric_pairs = config.get('asymmetric_pairs', [])
        categories = config.get('prompt_categories', ['identity'])
        
        runs = []
        
        for pair in asymmetric_pairs:
            for category in categories:
                prompts = PromptLibrary.get_by_category(category)
                for run_num in range(base_params.get('runs_per_pair', 1)):
                    for prompt in prompts[:1]:
                        run_id = f"{template_id}_{pair.get('label', 'pair')}_{category}_r{run_num}"
                        
                        runs.append(ExperimentRun(
                            run_id=run_id,
                            template_id=template_id,
                            parameters={
                                'agent_a_model': pair['agent_a'],
                                'agent_b_model': pair['agent_b'],
                                'max_turns': base_params.get('max_turns', 25),
                                'temperature': base_params.get('temperature', 0.7),
                                'run_number': run_num,
                                'pairing_label': pair.get('label')
                            },
                            prompt=prompt.prompt,
                            prompt_category=category
                        ))
        
        return runs
    
    def _generate_phenomenon_runs(self, template_id: str, template: Dict) -> List[ExperimentRun]:
        """Generate phenomenon-specific runs (identity leak, empathy, creativity)"""
        config = template['configuration']
        base_params = config['base_params']
        
        runs = []
        
        if 'specialized_prompts' in config:
            # Identity archaeology or similar
            for i, prompt in enumerate(config['specialized_prompts']):
                for run_num in range(base_params.get('runs_per_prompt', 1)):
                    run_id = f"{template_id}_p{i}_r{run_num}"
                    runs.append(ExperimentRun(
                        run_id=run_id,
                        template_id=template_id,
                        parameters={
                            'max_turns': base_params.get('max_turns', 30),
                            'agent_a_temp': base_params.get('agent_a_temp', 0.8),
                            'agent_b_temp': base_params.get('agent_b_temp', 0.8),
                            'run_number': run_num,
                            'prompt_index': i
                        },
                        prompt=prompt,
                        prompt_category='identity'
                    ))
        
        elif 'emotional_seed_prompts' in config:
            # Emotional contagion
            for i, emotion_config in enumerate(config['emotional_seed_prompts']):
                for run_num in range(base_params.get('runs_per_emotion', 1)):
                    run_id = f"{template_id}_{emotion_config['emotion']}_r{run_num}"
                    runs.append(ExperimentRun(
                        run_id=run_id,
                        template_id=template_id,
                        parameters={
                            'max_turns': base_params.get('max_turns', 25),
                            'agent_a_temp': base_params.get('agent_a_temp', 0.7),
                            'agent_b_temp': base_params.get('agent_b_temp', 0.7),
                            'run_number': run_num,
                            'emotion': emotion_config['emotion'],
                            'expected_trajectory': emotion_config.get('expected_trajectory')
                        },
                        prompt=emotion_config['prompt'],
                        prompt_category='emotional'
                    ))
        
        elif 'creative_prompts' in config:
            # Creativity emergence
            for i, prompt in enumerate(config['creative_prompts']):
                for run_num in range(base_params.get('runs_per_prompt', 1)):
                    run_id = f"{template_id}_creative{i}_r{run_num}"
                    runs.append(ExperimentRun(
                        run_id=run_id,
                        template_id=template_id,
                        parameters={
                            'max_turns': base_params.get('max_turns', 35),
                            'agent_a_temp': base_params.get('agent_a_temp', 1.0),
                            'agent_b_temp': base_params.get('agent_b_temp', 1.0),
                            'run_number': run_num,
                            'prompt_index': i
                        },
                        prompt=prompt,
                        prompt_category='creativity'
                    ))
        
        return runs
    
    def _generate_stress_test_runs(self, template_id: str, template: Dict) -> List[ExperimentRun]:
        """Generate stress test runs (breakdown, conflict, chaos)"""
        config = template['configuration']
        base_params = config['base_params']
        
        runs = []
        
        if 'stress_conditions' in config:
            # Breakdown cascade
            for i, condition in enumerate(config['stress_conditions']):
                for run_num in range(base_params.get('runs_per_condition', 1)):
                    run_id = f"{template_id}_{condition['type']}_r{run_num}"
                    
                    # Get a default prompt
                    prompts = PromptLibrary.get_by_category('problem_solving')
                    
                    runs.append(ExperimentRun(
                        run_id=run_id,
                        template_id=template_id,
                        parameters={
                            'max_turns': base_params.get('max_turns', 50),
                            'agent_a_temp': base_params.get('agent_a_temp', 0.7),
                            'agent_b_temp': base_params.get('agent_b_temp', 0.7),
                            'run_number': run_num,
                            'stress_condition': condition
                        },
                        prompt=prompts[0].prompt if prompts else "Let's have a conversation.",
                        prompt_category='problem_solving'
                    ))
        
        elif 'agent_configurations' in config:
            # Conflict escalation
            for agent_config in config['agent_configurations']:
                for topic in config.get('controversial_topics', []):
                    for run_num in range(base_params.get('runs_per_configuration', 1)):
                        run_id = f"{template_id}_{agent_config['label']}_{topic.get('category', 'topic')}_r{run_num}"
                        
                        runs.append(ExperimentRun(
                            run_id=run_id,
                            template_id=template_id,
                            parameters={
                                'max_turns': base_params.get('max_turns', 30),
                                'agent_a_temp': agent_config.get('agent_a_temp', 0.7),
                                'agent_b_temp': agent_config.get('agent_b_temp', 0.7),
                                'agent_a_system_prompt': agent_config.get('agent_a_system_prompt'),
                                'agent_b_system_prompt': agent_config.get('agent_b_system_prompt'),
                                'run_number': run_num,
                                'configuration_label': agent_config['label'],
                                'topic_category': topic.get('category')
                            },
                            prompt=topic['prompt'],
                            prompt_category='ethics'
                        ))
        
        elif 'chaos_prompts' in config:
            # Chaos injection
            for i, prompt in enumerate(config['chaos_prompts']):
                for run_num in range(base_params.get('runs_per_prompt', 1)):
                    run_id = f"{template_id}_chaos{i}_r{run_num}"
                    runs.append(ExperimentRun(
                        run_id=run_id,
                        template_id=template_id,
                        parameters={
                            'max_turns': base_params.get('max_turns', 20),
                            'agent_a_temp': base_params.get('agent_a_temp', 0.8),
                            'agent_b_temp': base_params.get('agent_b_temp', 0.8),
                            'run_number': run_num,
                            'prompt_index': i
                        },
                        prompt=prompt,
                        prompt_category='chaos'
                    ))
        
        return runs
    
    def _generate_longitudinal_runs(self, template_id: str, template: Dict) -> List[ExperimentRun]:
        """Generate longitudinal study runs"""
        config = template['configuration']
        base_params = config['base_params']
        
        runs = []
        
        if 'checkpoint_analysis' in config:
            # Ultra endurance
            categories = config.get('prompt_categories', ['problem_solving'])
            
            for category in categories:
                prompts = PromptLibrary.get_by_category(category)
                for run_num in range(base_params.get('runs_per_category', 1)):
                    for prompt in prompts[:1]:
                        run_id = f"{template_id}_{category}_r{run_num}"
                        runs.append(ExperimentRun(
                            run_id=run_id,
                            template_id=template_id,
                            parameters={
                                'max_turns': base_params.get('max_turns', 150),
                                'agent_a_temp': base_params.get('agent_a_temp', 0.7),
                                'agent_b_temp': base_params.get('agent_b_temp', 0.7),
                                'run_number': run_num,
                                'checkpoints': config['checkpoint_analysis']
                            },
                            prompt=prompt.prompt,
                            prompt_category=category
                        ))
        
        elif 'sessions' in base_params:
            # Multi-session consistency
            category = config.get('prompt_category', 'identity')
            prompts = PromptLibrary.get_by_category(category)
            
            for session_num in range(base_params['sessions']):
                run_id = f"{template_id}_session{session_num}"
                runs.append(ExperimentRun(
                    run_id=run_id,
                    template_id=template_id,
                    parameters={
                        'max_turns': base_params.get('turns_per_session', 15),
                        'agent_a_temp': base_params.get('agent_a_temp', 0.7),
                        'agent_b_temp': base_params.get('agent_b_temp', 0.7),
                        'session_number': session_num
                    },
                    prompt=prompts[session_num % len(prompts)].prompt if prompts else "Let's talk.",
                    prompt_category=category
                ))
        
        return runs
    
    def _generate_simple_runs(self, template_id: str, template: Dict) -> List[ExperimentRun]:
        """Generate simple runs for basic templates"""
        config = template['configuration']
        base_params = config['base_params']
        
        category = base_params.get('prompt_category', 'identity')
        prompts = PromptLibrary.get_by_category(category)
        
        runs = []
        for run_num in range(base_params.get('runs', 1)):
            for prompt in prompts[:1]:
                run_id = f"{template_id}_r{run_num}"
                runs.append(ExperimentRun(
                    run_id=run_id,
                    template_id=template_id,
                    parameters={
                        'max_turns': base_params.get('max_turns', 20),
                        'agent_a_temp': base_params.get('agent_a_temp', 0.7),
                        'agent_b_temp': base_params.get('agent_b_temp', 0.7),
                        'run_number': run_num
                    },
                    prompt=prompt.prompt,
                    prompt_category=category
                ))
        
        return runs
    
    def execute_run(self, run: ExperimentRun, verbose: bool = True) -> ExperimentRun:
        """
        Execute a single experiment run
        
        Args:
            run: ExperimentRun to execute
            verbose: Print progress
        
        Returns:
            Updated ExperimentRun with results
        """
        if verbose:
            print(f"  🔬 Executing: {run.run_id}")
        
        try:
            params = run.parameters
            
            # Create agents with custom parameters
            agent_a = None
            agent_b = None
            
            if 'agent_a_model' in params or 'agent_a_system_prompt' in params:
                agent_a = Agent(
                    role=AgentRole.AGENT_A,
                    model=params.get('agent_a_model', Config.AGENT_A_MODEL),
                    temperature=params.get('agent_a_temp', params.get('temperature', Config.AGENT_A_TEMPERATURE)),
                    system_prompt=params.get('agent_a_system_prompt', Config.AGENT_A_SYSTEM_PROMPT),
                    max_tokens=Config.AGENT_A_MAX_TOKENS
                )
            
            if 'agent_b_model' in params or 'agent_b_system_prompt' in params:
                agent_b = Agent(
                    role=AgentRole.AGENT_B,
                    model=params.get('agent_b_model', Config.AGENT_B_MODEL),
                    temperature=params.get('agent_b_temp', params.get('temperature', Config.AGENT_B_TEMPERATURE)),
                    system_prompt=params.get('agent_b_system_prompt', Config.AGENT_B_SYSTEM_PROMPT),
                    max_tokens=Config.AGENT_B_MAX_TOKENS
                )
            
            # Create conversation engine
            engine = ConversationEngine(
                seed_prompt=run.prompt,
                category=run.prompt_category,
                agent_a=agent_a,
                agent_b=agent_b,
                max_turns=params.get('max_turns', 20),
                context_window_size=params.get('context_window_size'),
                database=self.db
            )
            
            # Run conversation
            conversation = engine.run_conversation(blocking=True)
            
            run.conversation_id = conversation.id
            run.status = "completed"
            
            if verbose:
                print(f"    ✅ Completed: {conversation.total_turns} turns")
        
        except Exception as e:
            run.status = "error"
            run.error = str(e)
            if verbose:
                print(f"    ❌ Error: {e}")
        
        return run


def main():
    """CLI interface for template executor"""
    import sys
    
    executor = TemplateExecutor()
    
    if len(sys.argv) < 2:
        print("Usage: python template_executor.py <command> [args]")
        print("\nCommands:")
        print("  list [category]           - List available templates")
        print("  show <template_id>        - Show template details")
        print("  generate <template_id>    - Generate experiment runs")
        print("  execute <template_id>     - Execute template (limited runs)")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        category = sys.argv[2] if len(sys.argv) > 2 else None
        templates = executor.list_templates(category)
        
        print(f"\n📋 Available Templates" + (f" ({category})" if category else ""))
        print("=" * 80)
        
        for tmpl in templates:
            print(f"\n{tmpl['template_id']}")
            print(f"  Category: {tmpl['category']}")
            print(f"  Type: {tmpl['type']}")
            print(f"  Description: {tmpl['description']}")
            print(f"  Priority: {tmpl['priority']}")
            print(f"  Estimated Runs: {tmpl['estimated_runs']}")
    
    elif command == "show":
        if len(sys.argv) < 3:
            print("Usage: python template_executor.py show <template_id>")
            return
        
        template_id = sys.argv[2]
        template = executor.get_template(template_id)
        
        print(f"\n📋 Template: {template_id}")
        print("=" * 80)
        print(json.dumps(template, indent=2))
    
    elif command == "generate":
        if len(sys.argv) < 3:
            print("Usage: python template_executor.py generate <template_id>")
            return
        
        template_id = sys.argv[2]
        runs = executor.generate_experiment_runs(template_id)
        
        print(f"\n🔬 Generated {len(runs)} experiment runs for: {template_id}")
        print("=" * 80)
        
        for i, run in enumerate(runs[:10], 1):  # Show first 10
            print(f"\n{i}. {run.run_id}")
            print(f"   Parameters: {run.parameters}")
            print(f"   Prompt: {run.prompt[:80]}...")
        
        if len(runs) > 10:
            print(f"\n... and {len(runs) - 10} more runs")
    
    elif command == "execute":
        if len(sys.argv) < 3:
            print("Usage: python template_executor.py execute <template_id>")
            return
        
        template_id = sys.argv[2]
        runs = executor.generate_experiment_runs(template_id)
        
        print(f"\n🔬 Executing template: {template_id}")
        print(f"Total runs: {len(runs)}")
        print("=" * 80)
        
        # Limit execution for safety
        max_test_runs = 3
        if len(runs) > max_test_runs:
            print(f"\n⚠️  Limiting to {max_test_runs} runs for testing")
            print(f"Use batch_runner.py for full execution")
            runs = runs[:max_test_runs]
        
        for run in runs:
            executor.execute_run(run)


if __name__ == "__main__":
    main()
