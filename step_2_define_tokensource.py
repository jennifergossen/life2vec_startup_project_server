#!/usr/bin/env python3
# step2_define_tokensource.py - CORRECTED: Process BOTH startup events AND company attributes
import sys
import os
from pathlib import Path

# Add src to path
project_root = Path("/data/kebl8110/life2vec_startup_project")
sys.path.insert(0, str(project_root / "src"))

def main():
    print("🚀 Testing Life2Vec-Compatible CORRECTED Startup Data Processor")
    print("=" * 60)
    print("🔄 CORRECTED: Events without company description + Company static attributes")
    print("=" * 60)
    
    try:
        # Import BOTH processors
        from dataloaders.sources.startup_events import StartupEventsSource
        from dataloaders.sources.startup import StartupSource
        print("✅ Successfully imported BOTH sources")
        
        # Create BOTH processor instances
        startup_events = StartupEventsSource()
        startup_companies = StartupSource()
        print("✅ Created BOTH processor instances")
        
        # Check input files for BOTH
        print("\n📁 Checking input files...")
        
        # Check events data
        if startup_events.input_pickle.exists():
            size_mb = startup_events.input_pickle.stat().st_size / 1024 / 1024
            print(f"✅ Events file found: {size_mb:.1f} MB")
        else:
            print("❌ Events file not found!")
            return
        
        # Check company data
        if startup_companies.input_csv.exists():
            size_mb = startup_companies.input_csv.stat().st_size / 1024 / 1024
            print(f"✅ Company file found: {size_mb:.1f} MB")
        else:
            print("❌ Company file not found!")
            return
        
        # Show what will be processed
        print(f"\n📊 CORRECTED Processing Plan:")
        print(f"   🎯 Events: {len(startup_events.fields)} fields (NO company description)")
        print(f"   🏢 Companies: {len(startup_companies.fields)} fields (COUNTRY, CATEGORY, EMPLOYEE, DESCRIPTION)") 
        print(f"   📁 Output will go to:")
        print(f"      📦 Events: {project_root}/data/processed/sources/startup_events/")
        print(f"      🏢 Companies: {project_root}/data/processed/sources/startup/")
        print(f"\n🔄 LIFE2VEC METHODOLOGY:")
        print(f"   • Events = dynamic sequences")
        print(f"   • Company attributes = static background tokens")
        
        print("\n🚀 This will run:")
        print("   startup_events = StartupEventsSource()  # Events only")
        print("   startup_companies = StartupSource()     # Static company info")
        print("   startup_events.prepare()                # Process events")
        print("   startup_companies.prepare()             # Process company attributes")
        
        # Actually run if requested
        if "--run" in sys.argv:
            print("\n" + "=" * 60)
            print("🚀 STARTING CORRECTED STARTUP TOKENIZATION...")
            print("=" * 60)
            
            # Process events data (corrected - no company description)
            print("📦 Processing startup events data (CORRECTED)...")
            startup_events.prepare() 
            print("✅ Startup events tokenization complete!")
            
            # Process company data (static attributes)
            print("\n🏢 Processing startup company static data...")
            startup_companies.prepare()
            print("✅ Startup company tokenization complete!")
            
            print("\n🎉 CORRECTED TOKENIZATION COMPLETE!")
            print("✅ Events processed WITHOUT company description")
            print("✅ Company static attributes tokenized separately")
            print("🚀 Ready for combined corpus creation with proper Life2Vec methodology!")
            
        else:
            print("\n💡 To run the processing: python step2_define_tokensource.py --run")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()