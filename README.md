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
