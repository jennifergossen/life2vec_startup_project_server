#!/bin/bash
# Force complete rebuild by bypassing cache entirely

echo "🔥 FORCING COMPLETE REBUILD - BYPASSING CACHE"
echo "="*60

# 1. Delete ALL processed data
echo "🗑️ Deleting ALL processed data..."
rm -rf data/processed/sources/startup/
rm -rf data/processed/sources/startup_events/
rm -rf data/processed/corpus/

# 2. Temporarily change the decorator to force recompute
echo "🔧 Temporarily changing cache behavior..."

# Create a backup and modify the decorator parameter
cp src/dataloaders/sources/startup.py src/dataloaders/sources/startup.py.backup

# Change on_validation_error="recompute" to "error" to force rebuild
sed -i 's/on_validation_error="recompute"/on_validation_error="error"/g' src/dataloaders/sources/startup.py

echo "✅ Cache bypass applied"

# 3. Run tokenization
echo "🚀 Running tokenization with forced rebuild..."
CUDA_VISIBLE_DEVICES=3 python step_2_define_tokensource.py --run

# 4. Restore original file
echo "🔄 Restoring original startup.py..."
mv src/dataloaders/sources/startup.py.backup src/dataloaders/sources/startup.py

# 5. Verify results
echo "🔍 Verifying results..."
python -c "
import pandas as pd
import glob
files = glob.glob('data/processed/sources/startup/tokenized/*.parquet')
if files:
    df = pd.read_parquet(files[0])
    print('FORCED rebuild company IDs:')
    print(df.index[:5].tolist())
    print('Expected: [\"00000841...\", \"000014c8...\", ...]')
    expected = ['00000841-73a4-4c44-b713-34a26b7c6f99', '000014c8-63e8-4112-a6d7-50c89195874c']
    actual = df.index[:2].tolist()
    if actual == expected:
        print('✅ SUCCESS! IDs now match expected values!')
    else:
        print('❌ Still wrong IDs...')
else:
    print('❌ No tokenized files found')
"

echo "🎉 Forced rebuild complete!"
