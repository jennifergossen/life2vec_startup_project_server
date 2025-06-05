#!/usr/bin/env python3
"""
Life2Vec Step 1: Define Population
==================================
This script processes the startup population data, creating train/validation/test splits.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path("/data/kebl8110/life2vec_startup_project")
sys.path.insert(0, str(project_root / "src"))

def main():
    print("🚀 Life2Vec Step 1: Define Population")
    print("=" * 60)
    
    try:
        # Import the population class
        from dataloaders.populations.startups import StartupPopulation
        print("✅ Successfully imported StartupPopulation")
        
        # Create population instance
        startups = StartupPopulation()
        print("✅ Created StartupPopulation instance")
        
        # Check input files
        if startups.input_pickle.exists():
            size_mb = startups.input_pickle.stat().st_size / 1024 / 1024
            print(f"✅ Input pickle file found: {size_mb:.1f} MB")
        elif startups.input_csv.exists():
            size_mb = startups.input_csv.stat().st_size / 1024 / 1024
            print(f"✅ Input CSV file found: {size_mb:.1f} MB")
        else:
            print("❌ No input files found!")
            return
        
        print(f"📁 Output will go to: {project_root}/data/processed/populations/startups/")
        
        # Run if requested
        if "--run" in sys.argv:
            print("\n" + "=" * 60)
            print("🚀 STARTING STEP 1: POPULATION PROCESSING...")
            print("=" * 60)
            
            # Run the population preparation
            startups.prepare()
            
            print("\n🎉 STEP 1 COMPLETE!")
            print("✅ Population data prepared")
            print("✅ Train/Val/Test splits created")
            print("✅ Ready for Step 2: Define TokenSource")
        else:
            print("\n💡 To run: python step1_define_population.py --run")
            
    except Exception as e:
        print(f"❌ Error in Step 1: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
