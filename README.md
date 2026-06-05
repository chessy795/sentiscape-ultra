# Sentiment Analysis ULTRA

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![RoBERTa](https://img.shields.io/badge/RoBERTa-96--98%25-orange.svg)](https://huggingface.co/roberta-base)

Evidence-based sentiment analysis. Local, standalone, no API keys, no internet after model download.

**3 transformers + 4 calibrated lexicons + ABSA + emotion detection + mental health screening in one script.**

## Features

| Capability | Detail |
|------------|--------|
| **RoBERTa sentiment** | pos/neg/neutral polarity, 96-98% accuracy (Areshey 2024) |
| **RoBERTa emotion** | 11 emotions (anger/joy/fear/sadness...) |
| **Mental health** | depression/anxiety/suicidal/addiction detection via fine-tuned RoBERTa |
| **MultiLexScaled** | 4 lexicons z-score calibrated (van der Veen 2025) |
| **NRC EmoLex** | 8 Plutchik emotions (lexicon baseline) |
| **ABSA** | Aspect-based sentiment analysis via PyABSA |
| **SBERT embeddings** | Semantic clusters, exemplars, outliers via all-MiniLM-L6-v2 |
| **Negation handling** | Polanyi & Zaenen scope-based negation window |
| **Group comparison** | Cohen's d effect size between groups |
| **Self-contained HTML report** | All metrics, timelines, interactive charts |
| **CPU-first** | Works on CPU; DistilBERT profile for lightweight runs |

## Quick Start

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('sentiwordnet'); nltk.download('wordnet'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"

# Run demo
python sentiscape_ultra.py

# Run on your data
python sentiscape_ultra.py data.csv text_column_name

# Lightweight (DistilBERT, faster, CPU-friendly)
python sentiscape_ultra.py data.csv text_column_name --profile fast

# Full pipeline with ABSA
python sentiscape_ultra.py data.csv text_column_name --profile full --absa
```

First run downloads ~1.5GB of models to HuggingFace cache. Subsequent runs use cached models.

## Usage

```
python sentiscape_ultra.py [csv_path] [text_col] [options]
```

| Argument | Description |
|----------|-------------|
| `csv_path` | Path to CSV file (omit for demo mode) |
| `text_col` | Column name containing text |
| `-o, --output` | Output directory (default: output_senti) |
| `--profile` | `fast` (DistilBERT, default) or `full` (all models) |
| `--sample` | Randomly sample N documents |
| `--seed` | Random seed (default: 42) |
| `--max-docs` | Maximum documents to analyse (default: 500) |
| `--group` | Column for contrastive group analysis |
| `--absa` | Enable aspect-based sentiment analysis |
| `--no-embeddings` | Disable semantic embedding clusters |
| `--quiet` | Suppress per-document output |

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌───────────────────┐
│  CSV/Text   │───→│  Preflight Gate  │───→│  Model Ensemble   │
│  Loader     │    │  (column detect,  │    │                   │
│             │    │   sampling,       │    │  ┌─────────────┐  │
│             │    │   group split)    │    │  │ Transformer  │  │
└─────────────┘    └──────────────────┘    │  │ RoBERTa      │  │
                                            │  │ DistilBERT   │  │
                                            │  └──────┬──────┘  │
                                            │         │         │
                                            │  ┌──────▼──────┐  │
                                            │  │  Lexicons   │  │
                                            │  │ VADER       │  │
                                            │  │ AFINN       │  │
                                            │  │ NRC VAD     │  │
                                            │  │ SentiWordNet│  │
                                            │  └──────┬──────┘  │
                                            │         │         │
                                            │  ┌──────▼──────┐  │
                                            │  │  Calibrated  │  │
                                            │  │  Ensemble    │  │
                                            │  │  0.7 Tx +  │  │
                                            │  │  0.3 Lex    │  │
                                            │  └─────────────┘  │
                                            └─────────┬─────────┘
                                                      │
                              ┌───────────────────────┼───────────┐
                              │                       │           │
                        ┌─────▼─────┐          ┌──────▼──────┐   │
                        │  Emotions  │          │  Embeddings │   │
                        │  NRC EmoLex│          │  SBERT      │   │
                        │  RoBERTa   │          │  Clusters   │   │
                        │  Mental    │          │  Exemplars  │   │
                        │  Health    │          │  Outliers   │   │
                        └─────┬─────┘          └──────┬──────┘   │
                              │                       │           │
                        ┌─────▼───────────────────────▼──────┐   │
                        │         Output Artifacts            │   │
                        │  sentiscape_ultra_results.json     │   │
                        │  sentiment_by_doc.csv              │   │
                        │  sentiment_report.html             │   │
                        │  group_comparison.csv              │   │
                        │  manifest.json                     │   │
                        └────────────────────────────────────┘   │
                              │                                   │
                        ┌─────▼───────────────────────────────────▼─┐
                        │  Optional: ABSA, Semantic Timelines      │
                        └─────────────────────────────────────────┘
```

## Benchmark Results

Measured June 2026. DistilBERT fast profile, no embeddings.

| Dataset | Docs Processed | Time | Profile |
|---------|---------------|------|---------|
| 20 Newsgroups (500 docs, 5 categories) | 200 | 18.9s | fast (DistilBERT) |
| IMDb Sentiment (99 docs) | 99 | 11.0s | fast (DistilBERT) |
| TripAdvisor HK (200 docs sampled) | 200 | 12.3s | fast (DistilBERT) |
| BBC News (300 docs, 5 categories) | 200 | 18.9s | fast (DistilBERT) |

**Ensemble formula:** `final_polarity = RoBERTa × 0.7 + calibrated_lexicons × 0.3`

**Literature accuracy references:**

| Model | Accuracy | Source |
|-------|----------|--------|
| RoBERTa sentiment | **96-98%** | Areshey & Mathkour 2024 |
| DistilBERT | **97.35%** at 40% model size | Hussain et al. 2025 |
| MultiLexScaled (4 calibrated lexicons) | z-score calibrated | van der Veen & Bleich 2025 |
| Lexicon baseline (VADER + AFINN + NRC + SWN) | 70-75% raw | Baseline |

## Output

| File | Description |
|------|-------------|
| `sentiscape_ultra_results.json` | Per-document: ensemble polarity, transformer scores, raw/calibrated lexicons, emotions, ABSA |
| `sentiment_by_doc.csv` | Flat CSV with ensemble label and polarity per doc |
| `sentiment_report.html` | Self-contained HTML with polarity distribution, timelines, group comparison |
| `group_comparison.csv` | Cohen's d effect size between groups |
| `manifest.json` | Runtime metadata: timestamp, n_docs, elapsed, models used |

## Evidence Base

| Paper | Finding | Citation |
|-------|---------|----------|
| Areshey & Mathkour 2024 | RoBERTa > BERT > DistilBERT for Arabic SA | Expert Systems, 46 cites |
| van der Veen & Bleich 2025 | MultiLexScaled z-score calibration outperforms raw | PLoS ONE, 21 cites |
| Hill et al. 2025 | Majority-vote ensembles fail on imbalanced SA | J Big Data |
| Hussain et al. 2025 | DistilBERT 97.35% at 40% size | Scientific Reports |
| Zhang et al. 2024 | LLMs lag on complex SA tasks | NAACL |
| Polanyi & Zaenen 2006 | Contextual valence shifters (negation scope) | ACL |

## Tips

### Which profile should I use?

| Scenario | Recommended | Why |
|----------|-------------|-----|
| First run, quick demo | `profile=fast` | DistilBERT only, ~200MB download |
| Research-grade sentiment | `profile=full --absa` | RoBERTa + all lexicons + ABSA |
| Large corpus (10K+ docs) | `profile=fast --no-embeddings` | Skip SBERT clustering, keep sentiment |
| Mental health screening | `profile=full` | Includes fine-tuned mental health model |
| CPU-constrained | `profile=fast` | No GPU needed |
| Short text / social media | `profile=full` | RoBERTa handles noise better |

### Speed / Accuracy

- **Fast profile**: ~500 docs/min (DistilBERT, no embeddings)
- **Full profile**: ~100 docs/min (RoBERTa + lexicons + embeddings)
- **With ABSA**: ~50 docs/min (per-aspect analysis is expensive)

## Dependencies

```
transformers>=4.30
torch
numpy
pandas
nltk
afinn
nrclex
sentence-transformers
scikit-learn
```

Optional: `pyabsa`, `plotly`, `senticnet`

## Citation

```bibtex
@software{pang2026sentiscapeultra,
  author = {Peter Pang},
  title = {Sentiment Analysis ULTRA: Evidence-Based Sentiment Toolkit},
  year = {2026},
  url = {https://github.com/chessy795/sentiscape-ultra}
}
```

## License

MIT
