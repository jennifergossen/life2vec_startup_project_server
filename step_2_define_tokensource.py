#!/usr/bin/env python3
# step2_define_tokensource.py - Test your life2vec-compatible processor
import sys
import os
from pathlib import Path

# Add src to path
project_root = Path("/data/kebl8110/life2vec_startup_project")
sys.path.insert(0, str(project_root / "src"))

def main():
    print("�� Testing Life2Vec-Compatible Startup Events Processor")
    print("=" * 60)
    
    try:
        # Import your processor
        from dataloaders.sources.startup_events import StartupEventsSource
        print("✅ Successfully imported StartupEventsSource")
        
        # Create processor instance
        startup_events = StartupEventsSource()
        print("✅ Created processor instance")
        
        # Check input file
        if startup_events.input_pickle.exists():
            size_mb = startup_events.input_pickle.stat().st_size / 1024 / 1024
            print(f"✅ Input file found: {size_mb:.1f} MB")
        else:
            print("❌ Input file not found!")
            return
        
        # Show what will be processed
        print(f"📊 Will process {len(startup_events.fields)} fields")
        print(f"📁 Output will go to: {project_root}/data/processed/sources/startup_events/")
        
        print("\n🚀 This will run:")
        print("   startup_events = StartupEventsSource()")
        print("   startup_events.prepare()")
        
        # Actually run if requested
        if "--run" in sys.argv:
            print("\n" + "=" * 60)
            print("🚀 STARTING STARTUP TOKENIZATION...")
            print("=" * 60)
            
            # This processes your startup events data
            print("Preparing startup events tokenization...")
            startup_events.prepare() 
            print("✅ Startup events tokenization complete!")
            
            print("\n🎉 TOKENIZATION COMPLETE!")
            
        else:
            print("\n💡 To run the processing: python step2_define_tokensource.py --run")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
