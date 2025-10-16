{{ ... }}
db = Database(Config.DATABASE_PATH)
evaluator = TemplateEvaluator()
llm_analyzer = LLMIdentityAnalyzer()  # Will use Config.ANALYSIS_MODEL

print("\n" + "="*80)
print("PILOT FACTORIAL: LLM-BASED RE-ANALYSIS")
print("="*80 + "\n")

print(f"🤖 Analysis model: {llm_analyzer.model}")
print(f"⚠️  This will call the model ~180 times (30 messages × 6 conversations)")
{{ ... }}
