#!/bin/bash

# Create the complete life2vec startup project structure
echo "🏗️ Creating life2vec startup project structure..."

# Main data directories
mkdir -p data/{cleaned/cleaned_startup,interim/{corpus/startup,vocab/startup/token_counts/startup},processed/{corpus,datasets},processed_startup,rawdata_startup}

# Main source directories with __pycache__ placeholders
mkdir -p src/{dataloaders/{populations,preprocessing,sources,tasks},models,transformer}

# Create __pycache__ directories (we'll add .gitkeep files to track them)
mkdir -p __pycache__
mkdir -p src/__pycache__
mkdir -p src/dataloaders/__pycache__
mkdir -p src/dataloaders/populations/__pycache__
mkdir -p src/dataloaders/preprocessing/__pycache__
mkdir -p src/dataloaders/sources/__pycache__
mkdir -p src/dataloaders/tasks/__pycache__
mkdir -p src/models/__pycache__
mkdir -p src/transformer/__pycache__

# Create __init__.py files for Python packages
touch src/__init__.py
touch src/dataloaders/__init__.py
touch src/dataloaders/populations/__init__.py
touch src/dataloaders/preprocessing/__init__.py
touch src/dataloaders/sources/__init__.py
touch src/dataloaders/tasks/__init__.py
touch src/models/__init__.py
touch src/transformer/__init__.py

# Create .gitkeep files for empty directories that need to be tracked
touch data/cleaned/cleaned_startup/.gitkeep
touch data/interim/corpus/startup/.gitkeep
touch data/interim/vocab/startup/token_counts/startup/.gitkeep
touch data/processed/corpus/.gitkeep
touch data/processed/datasets/.gitkeep
touch data/processed_startup/.gitkeep
touch data/rawdata_startup/.gitkeep

# Create placeholder files that we'll need to populate
touch src/dataloaders/populations/base.py
touch src/dataloaders/populations/startup.py
touch src/dataloaders/populations/users.py

touch src/dataloaders/preprocessing/startup_cleaner.py

touch src/dataloaders/sources/base.py
touch src/dataloaders/sources/startup_events.py
touch src/dataloaders/sources/startup.py

touch src/dataloaders/tasks/base.py
touch src/dataloaders/tasks/pretrain_startup.py
touch src/dataloaders/tasks/pretrain.py
touch src/dataloaders/tasks/startup_tasks.py

touch src/dataloaders/augment.py
touch src/dataloaders/datamodule_startup.py
touch src/dataloaders/datamodule.py
touch src/dataloaders/dataset.py
touch src/dataloaders/decorators.py
touch src/dataloaders/ops.py
touch src/dataloaders/pretrain.py
touch src/dataloaders/serialize.py
touch src/dataloaders/types.py
touch src/dataloaders/utils.py
touch src/dataloaders/vocabulary_startup.py
touch src/dataloaders/vocabulary.py

touch src/models/pretrain.py

touch src/transformer/attention.py
touch src/transformer/embeddings.py
touch src/transformer/modules.py
touch src/transformer/performer.py
touch src/transformer/transformer_utils.py
touch src/transformer/transformer.py

touch src/utils.py

echo "✅ Directory structure created!"

# Create project files
echo "📄 Creating project files..."

# Create .gitignore
cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
pdm.lock
.pdm.toml

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
.idea/

# Data directories (but keep structure)
data/interim/
data/processed/
!data/interim/.gitkeep
!data/processed/.gitkeep

# Model checkpoints and logs
checkpoints/
checkpoints_beast/
lightning_logs/
logs/
wandb/

# Temporary files
*.tmp
*.temp
.DS_Store
Thumbs.db
EOF

# Create LICENSE (MIT)
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Life2vec Startup Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# Create README.md
cat > README.md << 'EOF'
# Life2vec Startup Project

A transformer-based model for predicting startup outcomes using life event sequences.

## Project Structure

```
life2vec_startup_project/
├── data/
│   ├── cleaned/cleaned_startup/     # Cleaned datasets
│   ├── interim/                     # Intermediate processing files
│   ├── processed/                   # Final processed data
│   ├── processed_startup/           # Startup-specific processed data
│   └── rawdata_startup/             # Raw startup data
├── src/
│   ├── dataloaders/                 # Data loading and processing
│   ├── models/                      # Model definitions
│   ├── transformer/                 # Transformer architecture
│   └── utils.py                     # Utility functions
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
- Place raw data in `data/rawdata_startup/`
- Run data cleaning scripts
- Process data through the pipeline

3. Train the model:
```bash
python step_5_train_startup2vec.py
```

## Data Files Expected

### In `data/cleaned/cleaned_startup/`:
- `combined_events_cleaned.csv`
- `combined_events_cleaned.pkl`
- `company_base_cleaned.csv`
- `company_base_cleaned.pkl`

### In `data/processed_startup/`:
- `combined_events.csv`
- `combined_events.pkl`
- `company_base.csv`
- `company_base.pkl`

## Model Architecture

Based on the life2vec transformer architecture adapted for startup event sequences.

## License

MIT License - see LICENSE file for details.
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
torch>=2.0.0
pytorch-lightning>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
dask>=2024.1.0
matplotlib>=3.7.0
h5py>=3.9.0
performer-pytorch>=1.1.4
ipykernel>=6.25.0
pyarrow>=12.0.0
torchmetrics>=1.0.0
wandb>=0.15.0
scikit-learn>=1.3.0
seaborn>=0.12.0
tqdm>=4.65.0
joblib>=1.3.0
EOF

echo "✅ Project files created!"

# Show the structure
echo "🌳 Project structure:"
tree . || find . -type d | head -20

echo ""
echo "🚀 Next steps:"
echo "1. Run: git add ."
echo "2. Run: git commit -m 'Initial project structure'"
echo "3. Create GitHub repo and push"
echo "4. Start populating the Python files"
EOF
