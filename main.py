#!/usr/bin/env python3
"""
AA Microscope - Agent-Agent Conversation Observatory
Main entry point

Launch with: python main.py
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from interface.tui import launch_tui
from config import Config


def main():
    """Main entry point"""
    print("🔬 AA Microscope - Agent-Agent Conversation Observatory")
    print("=" * 60)
    
    # Validate configuration
    if not Config.validate():
        print("\n❌ Configuration Error!")
        print("Please create a .env file with your API keys.")
        print("See .env.example for template.")
        sys.exit(1)
    
    print("\n✅ Configuration valid")
    print("\nLaunching terminal interface...\n")
    
    # Launch TUI
    try:
        launch_tui()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
