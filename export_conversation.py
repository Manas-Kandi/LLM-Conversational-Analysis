#!/usr/bin/env python3
"""
Quick script to export conversations in friendly formats
"""
import sys
from pathlib import Path
from storage.database import Database
from storage.export import ConversationExporter
from config import Config

def main():
    if len(sys.argv) < 2:
        print("Usage: python export_conversation.py <conversation_id> [format]")
        print("Formats: markdown, html, json, all (default: all)")
        print("\nExample: python export_conversation.py 1 html")
        sys.exit(1)
    
    conversation_id = int(sys.argv[1])
    format_type = sys.argv[2] if len(sys.argv) > 2 else "all"
    
    # Initialize
    db = Database(Config.DATABASE_PATH)
    exporter = ConversationExporter(db)
    
    print(f"🔬 Exporting conversation #{conversation_id}...")
    
    try:
        if format_type == "all":
            results = exporter.export_all_formats(conversation_id)
            print("\n✅ Exported to:")
            for fmt, path in results.items():
                print(f"  📄 {fmt.upper()}: {path}")
        
        elif format_type == "markdown" or format_type == "md":
            from config import EXPORTS_DIR
            output_path = Path(EXPORTS_DIR) / f"conversation_{conversation_id}.md"
            exporter.export_to_markdown(conversation_id, output_path)
            print(f"\n✅ Exported to: {output_path}")
        
        elif format_type == "html":
            from config import EXPORTS_DIR
            output_path = Path(EXPORTS_DIR) / f"conversation_{conversation_id}.html"
            exporter.export_to_html(conversation_id, output_path)
            print(f"\n✅ Exported to: {output_path}")
            print(f"💡 Open in browser: file://{output_path.absolute()}")
        
        elif format_type == "json":
            from config import EXPORTS_DIR
            output_path = Path(EXPORTS_DIR) / f"conversation_{conversation_id}.json"
            exporter.export_to_json(conversation_id, output_path)
            print(f"\n✅ Exported to: {output_path}")
        
        else:
            print(f"❌ Unknown format: {format_type}")
            print("Available formats: markdown, html, json, all")
            sys.exit(1)
        
        print("\n🎉 Export complete!")
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
