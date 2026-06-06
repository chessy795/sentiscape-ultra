#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SENTIMENT ANALYSIS ULTRA v3 — Evidence-Based Sentiment Analysis
================================================================
Local, standalone, no API keys, no internet after model download.

3 transformers + 4 calibrated lexicons + ABSA + emotion detection.

Usage:
    python sentiscape_ultra.py                             # demo
    python sentiscape_ultra.py data.csv text_col           # your data
    python sentiscape_ultra.py data.csv text_col --profile fast
    python sentiscape_ultra.py data.csv text_col --profile full --absa
    python sentiscape_ultra.py data.csv text_col --sample 100 --quiet -o results

First run downloads ~1.5GB of models. Subsequent runs use cached models.
"""
from __future__ import annotations

import argparse, json, os, re, sys, time, warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from embeddings_util import semantic_profile
except Exception:
    semantic_profile = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# --- Shared infrastructure (ultra_shared, optional) ---
_ultra_parent = str(Path(__file__).resolve().parent.parent)
if _ultra_parent not in sys.path:
    sys.path.insert(0, _ultra_parent)

try:
    from ultra_shared.logging import setup_logging as _setup_logging
    from ultra_shared.config import load_config as _load_config
    from ultra_shared.data import load_documents as _load_documents
    HAS_ULTRA_SHARED = True
except ImportError:
    HAS_ULTRA_SHARED = False

try:
    from ultra_shared.schema import build_manifest, new_doc, add_tool_section, write_docs_jsonl
    from ultra_shared.schema import write_manifest as _write_manifest
    HAS_SCHEMA = True
except ImportError:
    HAS_SCHEMA = False

try:
    from ultra_shared.report import ReportBuilder, THRESHOLDS
    HAS_REPORT = True
except ImportError:
    HAS_REPORT = False

NRC_VAD_PATH = Path(__file__).resolve().parent / "data" / "NRC-VAD-Lexicon-v2.1.txt"
PLUTCHIK_EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "nowhere", "nothing",
    "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't",
    "doesn't", "don't", "didn't", "won't", "wouldn't", "can't", "cannot",
    "couldn't", "shouldn't", "mustn't", "needn't", "daren't", "mightn't",
    "n't",  # matches contractions without apostrophe normalization
}
NEGATION_WINDOW = 3  # negate next 3 content words


def _apply_negation(tokens, scores_per_token):
    """Adjust token-level scores for negation.

    For each negation word, negate the scores of the next NEGATION_WINDOW
    content words by multiplying by -1. Based on Polanyi & Zaenen (2006).
    """
    adjusted = list(scores_per_token)
    for i, token in enumerate(tokens):
        if token.lower() in NEGATION_WORDS or token.lower().endswith("n't"):
            for j in range(i + 1, min(i + 1 + NEGATION_WINDOW, len(tokens))):
                if tokens[j].isalpha():
                    adjusted[j] = -adjusted[j]
    return adjusted


class Lexicons:
    def __init__(self):
        self._cache = {}

    def vader(self, text: str) -> float:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            try:
                nltk.data.find("sentiment/vader_lexicon")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
            if "v" not in self._cache:
                self._cache["v"] = SentimentIntensityAnalyzer()
            return self._cache["v"].polarity_scores(text)["compound"]
        except Exception:
            return 0.0

    def afinn(self, text: str) -> float:
        try:
            from afinn import Afinn
            if "a" not in self._cache:
                self._cache["a"] = Afinn()
            words = re.findall(r"[A-Za-z']+", text.lower())
            word_scores = [self._cache["a"].score(w) for w in words]
            adjusted = _apply_negation(words, word_scores)
            return float(np.mean(adjusted)) if adjusted else 0.0
        except Exception:
            return 0.0

    def nrc_vad(self, text: str) -> float:
        try:
            if "vad" not in self._cache:
                if NRC_VAD_PATH.exists():
                    df = pd.read_csv(NRC_VAD_PATH, sep="\t")
                    self._cache["vad"] = dict(zip(df["term"].str.lower(), df["valence"]))
                else:
                    return 0.0
            words = re.findall(r"[A-Za-z']+", text.lower())
            vals = [self._cache["vad"].get(w, 0.5) for w in words]
            # Apply negation: for VAD, valence 0-1, negated = 1 - valence
            adjusted = []
            for i, w in enumerate(words):
                v = self._cache["vad"].get(w, 0.5)
                adjusted.append(v)
            adjusted = _apply_negation(words, [a - 0.5 for a in adjusted])
            adjusted = [a + 0.5 for a in adjusted]  # shift back to 0-1 range
            return float(np.mean([a for a in adjusted if 0 <= a <= 1])) if adjusted else 0.0
        except Exception:
            return 0.0

    def swn(self, text: str) -> float:
        try:
            from nltk.corpus import sentiwordnet as swn_mod
            from nltk import pos_tag, word_tokenize
            from nltk.corpus import wordnet as wn

            def wn_pos(tag):
                if tag.startswith("J"): return wn.ADJ
                if tag.startswith("V"): return wn.VERB
                if tag.startswith("N"): return wn.NOUN
                if tag.startswith("R"): return wn.ADV
                return None

            pols = []
            for word, tag in pos_tag(word_tokenize(text)):
                p = wn_pos(tag)
                if p is None: continue
                syns = list(swn_mod.senti_synsets(word, p))[:3]
                if syns:
                    pols.append(np.mean([s.pos_score() - s.neg_score() for s in syns]))
            return float(np.mean(pols)) if pols else 0.0
        except Exception:
            return 0.0

    def nrc_emotions(self, text: str) -> Dict[str, float]:
        try:
            from nrclex import NRCLex
            lex = NRCLex(text)
            nrc_dict = lex.__lexicon__ if hasattr(lex, "__lexicon__") else {}
            if not nrc_dict: return {}
            words = re.findall(r"[A-Za-z']+", text.lower())
            counts = {}
            for w in words:
                if w in nrc_dict:
                    for emo in nrc_dict[w]:
                        counts[emo] = counts.get(emo, 0) + 1
            n = max(len(words), 1)
            return {emo: counts.get(emo, 0) / n for emo in PLUTCHIK_EMOTIONS}
        except Exception:
            return {}


class MultiLexScaled:
    def __init__(self):
        self._raw = {"vader": [], "afinn": [], "nrc_vad": [], "swn": []}
        self.stats = None

    def collect(self, name: str, score: float):
        self._raw.setdefault(name, []).append(score)

    def fit(self):
        self.stats = {}
        for name, scores in self._raw.items():
            arr = np.array(scores)
            self.stats[name] = {"mean": float(np.mean(arr)), "std": float(np.std(arr)) if len(arr) > 1 else 1.0}

    def calibrate(self, name: str, raw: float) -> float:
        if self.stats is None: return raw
        s = self.stats[name]
        return (raw - s["mean"]) / s["std"] if s["std"] > 1e-12 else 0.0

    def ensemble(self, scores: Dict[str, float]) -> float:
        z = [self.calibrate(n, v) for n, v in scores.items()]
        return float(np.mean(z)) if z else 0.0


class TransformerEngine:
    MODELS = {
        "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "emotion": "cardiffnlp/twitter-roberta-base-emotion-latest",
        "goemotions": "bhadresh-savani/roberta-base-go-emotions",
        "lightweight": "distilbert-base-uncased-finetuned-sst-2-english",
    }

    def __init__(self, tier: str = "full"):
        self._tier = tier
        self._pipes = {}

    def _load(self, key: str):
        if key in self._pipes: return
        if key not in self.MODELS: return
        try:
            from transformers import pipeline
            task = "sentiment-analysis" if key in ("sentiment", "lightweight") else "text-classification"
            self._pipes[key] = pipeline(task, model=self.MODELS[key], top_k=None, truncation=True)
        except Exception as e:
            print(f"  [WARN] Failed to load {key}: {e}")

    def score(self, text: str) -> Dict[str, Any]:
        results = {}
        if self._tier == "lightweight":
            self._load("lightweight")
            r = self._run("lightweight", text)
            if r: results["lightweight"] = r
        else:
            for key in ("sentiment", "emotion", "goemotions"):
                self._load(key)
                r = self._run(key, text)
                if r: results[key] = r
        return self._combine(results)

    def _run(self, key: str, text: str) -> Dict:
        if key not in self._pipes: return {}
        try:
            raw = self._pipes[key](text[:512])
            if raw and isinstance(raw[0], list): raw = raw[0]
            scores = {r["label"].lower(): r["score"] for r in raw}
            if key in ("sentiment", "lightweight"):
                pos = scores.get("positive", scores.get("pos", 0))
                neg = scores.get("negative", scores.get("neg", 0))
                label = max(scores, key=scores.get)
                return {"polarity": round(pos - neg, 4), "scores": scores, "label": label,
                        "confidence": round(scores[label], 4), "model": self.MODELS[key]}
            elif key == "emotion":
                top = max(scores, key=scores.get)
                return {"scores": scores, "top_emotion": top, "confidence": round(scores[top], 4),
                        "model": self.MODELS[key]}
            elif key == "goemotions":
                top = max(scores, key=scores.get)
                return {"scores": scores, "top_emotion": top, "confidence": round(scores[top], 4),
                        "model": self.MODELS[key]}
        except Exception as e:
            return {"error": str(e)}
        return {}

    def _combine(self, results: Dict) -> Dict:
        polarity = 0
        if "sentiment" in results:
            polarity = results["sentiment"]["polarity"]
        elif "lightweight" in results:
            polarity = results["lightweight"]["polarity"]
        return {
            "transformer_polarity": round(polarity, 4),
            "transformer_label": "positive" if polarity > 0.05 else "negative" if polarity < -0.05 else "neutral",
            "results": results, "available": bool(results),
        }


class ABSA:
    def __init__(self):
        self._extractor = None

    def extract(self, text: str) -> Dict[str, Any]:
        try:
            from pyabsa import AspectTermExtraction as ATE
            if self._extractor is None:
                self._extractor = ATE.AspectExtractor("multilingual", auto_device=True)
            result = self._extractor.predict([text])
            if not result: return {"aspects": [], "available": True}
            r = result[0] if isinstance(result, list) else result
            aspects = []
            for key in ("aspect", "opinion", "sentiment", "probability"):
                if not isinstance(r.get(key), list): r[key] = [r[key]] if r.get(key) else []
            for i in range(len(r.get("aspect", []))):
                asp = r["aspect"][i]
                if asp:
                    aspects.append({"aspect": asp,
                                    "opinion": r.get("opinion", [])[i] if i < len(r.get("opinion", [])) else "",
                                    "sentiment": r.get("sentiment", [])[i] if i < len(r.get("sentiment", [])) else "",
                                    "confidence": round(float(r.get("probability", [0])[i] if i < len(r.get("probability", [])) else 0), 4)})
            return {"aspects": aspects, "n_aspects": len(aspects), "available": True}
        except Exception as e:
            return {"aspects": [], "available": False, "error": str(e)}


class SentimentEnsemble:
    def __init__(self, tier: str = "full"):
        self.lex = Lexicons()
        self.transformer = TransformerEngine(tier)
        self.mls = MultiLexScaled()
        self.absa = ABSA()
        self._tier = tier

    def _pass1_calibrate(self, texts: List[str]):
        print("  [Pass 1] Lexicon calibration...")
        for i, text in enumerate(texts):
            self.mls.collect("vader", self.lex.vader(text))
            self.mls.collect("afinn", self.lex.afinn(text))
            self.mls.collect("nrc_vad", self.lex.nrc_vad(text))
            self.mls.collect("swn", self.lex.swn(text))
            if (i + 1) % 100 == 0:
                print(f"    ...{i+1}/{len(texts)}")
        self.mls.fit()
        print(f"  Calibration done ({len(texts)} docs)")

    def analyze_text(self, text: str, doc_id: str = "", calibrate: bool = True) -> Dict[str, Any]:
        transformer = self.transformer.score(text)
        raw = {"vader": self.lex.vader(text), "afinn": self.lex.afinn(text),
               "nrc_vad": self.lex.nrc_vad(text), "swn": self.lex.swn(text)}
        lex_cal = self.mls.ensemble(raw) if calibrate else float(np.mean(list(raw.values())))
        nrc_emo = self.lex.nrc_emotions(text)

        if transformer.get("available"):
            tr_conf = abs(transformer["transformer_polarity"])
            tr_weight = min(0.5 + tr_conf * 0.5, 0.85)
            final = transformer["transformer_polarity"] * tr_weight + lex_cal * (1 - tr_weight)
            n_src = 5
        else:
            final = lex_cal
            n_src = 4

        label = "positive" if final > 0.05 else "negative" if final < -0.05 else "neutral"
        tr_res = transformer.get("results", {})

        return {
            "doc_id": doc_id, "text_preview": text[:200],
            "ensemble_polarity": round(final, 4), "ensemble_label": label, "n_sources": n_src,
            "transformer": transformer, "lexicon_raw": raw, "lexicon_calibrated": round(lex_cal, 4),
            "nrc_emotions": nrc_emo,
            "roberta_emotions": tr_res.get("emotion", {}).get("scores", {}),
            "goemotions_scores": tr_res.get("goemotions", {}).get("scores", {}),
        }

    def analyze_documents(self, df: pd.DataFrame, text_col: str = "clean_text",
                          id_col: str = "doc_id", group_col: str = None,
                          max_docs: int = 5000, run_absa: bool = False,
                          run_embeddings: bool = True) -> Dict[str, Any]:
        if len(df) > max_docs:
            print(f"  [!] Truncating to first {max_docs} docs (corpus has {len(df)} total)")
        df = df.head(max_docs).copy()
        texts = df[text_col].fillna("").astype(str).tolist()
        ids = df[id_col].astype(str).tolist()

        if len(texts) < 30:
            print("  [!] WARNING: <30 docs for calibration — z-scores will be unstable")
        self._pass1_calibrate(texts)

        print("  [Pass 2] Analyzing documents...")
        results = []
        t0 = time.time()
        for i, (text, doc_id) in enumerate(zip(texts, ids)):
            if not text.strip(): continue
            r = self.analyze_text(text, doc_id=doc_id, calibrate=True)
            if group_col and group_col in df.columns:
                r["group"] = df.iloc[i].get(group_col, "")
            results.append(r)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(texts) - i - 1)
                print(f"    ...{i+1}/{len(texts)}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
        elapsed = time.time() - t0
        print(f"  Done: {len(results)} docs in {elapsed:.1f}s ({len(results)/max(elapsed,0.1):.0f} docs/sec)")

        absa_results = []
        if run_absa:
            print("  [Pass 3] ABSA...")
            for i, r in enumerate(results):
                text = texts[i] if i < len(texts) else ""
                if text.strip():
                    absa_r = self.absa.extract(text)
                    absa_results.append(absa_r)
                    r["aspects"] = absa_r.get("aspects", [])
                if (i + 1) % 50 == 0:
                    print(f"    ...{i+1}/{len(results)}")

        polarities = [r["ensemble_polarity"] for r in results]
        labels = [r["ensemble_label"] for r in results]

        summary = {
            "doc_count": len(results),
            "mean_polarity": round(float(np.mean(polarities)), 4) if polarities else 0,
            "median_polarity": round(float(np.median(polarities)), 4) if polarities else 0,
            "std_polarity": round(float(np.std(polarities)), 4) if polarities else 0,
            "label_distribution": dict(Counter(labels)),
            "calibration_stats": self.mls.stats,
        }

        semantic = {"available": False, "reason": "disabled"}
        if run_embeddings and semantic_profile is not None:
            print("  [Pass 3] Semantic embeddings + cluster profile...")
            semantic = semantic_profile(
                texts=texts[:len(results)],
                polarities=[r["ensemble_polarity"] for r in results],
                labels=[r["ensemble_label"] for r in results],
                ids=[r["doc_id"] for r in results],
                cache_dir=Path(__file__).resolve().parent / "cache",
            )
        elif run_embeddings:
            semantic = {"available": False, "reason": "embeddings_util import failed"}

        group_summary = {}
        if group_col:
            groups = {}
            for r in results:
                g = r.get("group", "unknown")
                groups.setdefault(g, []).append(r)
            for g, docs in groups.items():
                g_pols = [d["ensemble_polarity"] for d in docs]
                group_summary[g] = {
                    "count": len(docs),
                    "mean_polarity": round(float(np.mean(g_pols)), 4),
                    "std_polarity": round(float(np.std(g_pols)), 4),
                    "label_dist": dict(Counter(d["ensemble_label"] for d in docs)),
                }
            if len(groups) >= 2:
                g1, g2 = list(groups.keys())[:2]
                g1_pols = [d["ensemble_polarity"] for d in groups[g1]]
                g2_pols = [d["ensemble_polarity"] for d in groups[g2]]
                pooled_std = np.sqrt((np.var(g1_pols) + np.var(g2_pols)) / 2)
                cohens_d = (np.mean(g1_pols) - np.mean(g2_pols)) / pooled_std if pooled_std > 0 else 0
                group_summary["effect_size"] = {
                    "cohens_d": round(float(cohens_d), 4), "groups": [g1, g2],
                    "interpretation": "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small",
                }

        return {"documents": results, "summary": summary, "group_summary": group_summary,
                "semantic_profile": semantic,
                "absa_results": absa_results, "total_documents": len(results)}


# ─── S5: Emoji Integration ──────────────────────────────────────────────────

# Emoji → sentiment polarity mapping (subset of Unicode CLDR + common emojis)
EMOJI_SENTIMENT = {
    "😀": 0.8, "😃": 0.8, "😄": 0.8, "😁": 0.7, "😆": 0.7, "😅": 0.5,
    "🤣": 0.6, "😂": 0.6, "🙂": 0.5, "🙃": 0.2, "😉": 0.4, "😊": 0.8,
    "😇": 0.7, "🥰": 0.9, "😍": 0.9, "🤩": 0.9, "😘": 0.8, "😗": 0.6,
    "😚": 0.7, "😙": 0.6, "🥲": 0.3, "😋": 0.7, "😛": 0.5, "😜": 0.5,
    "🤪": 0.5, "😝": 0.5, "🤑": 0.4, "🤗": 0.7, "🤭": 0.4, "🫢": 0.3,
    "🤫": 0.2, "🤔": 0.1, "🫡": 0.4, "🤐": 0.0, "🫠": 0.3,
    "😐": 0.0, "😑": -0.1, "😶": 0.0, "😏": 0.3, "😒": -0.3,
    "🙄": -0.3, "😬": -0.2, "🤥": -0.3, "😌": 0.4, "😔": -0.4,
    "😪": -0.3, "🤤": 0.3, "😴": 0.0, "😷": -0.2, "🤒": -0.3,
    "🤕": -0.4, "🤢": -0.5, "🤮": -0.8, "🥵": -0.3, "🥶": -0.2,
    "🥴": -0.2, "😵": -0.3, "🤯": -0.1, "🥳": 0.8, "🥸": 0.3,
    "😎": 0.6, "🤓": 0.4, "🧐": 0.2,
    "😕": -0.2, "🫤": -0.1, "😟": -0.4, "🙁": -0.4, "☹️": -0.5,
    "😮": 0.1, "😯": 0.1, "😲": 0.2, "😳": 0.0, "🥺": -0.1,
    "🥹": 0.1, "😦": -0.3, "😧": -0.3, "😨": -0.6, "😰": -0.5,
    "😥": -0.4, "😢": -0.6, "😭": -0.7, "😱": -0.7, "😖": -0.5,
    "😣": -0.4, "😞": -0.5, "😓": -0.4, "😩": -0.5, "😫": -0.5,
    "🥱": -0.2, "😤": -0.4, "😡": -0.8, "😠": -0.7, "🤬": -0.9,
    "❤️": 0.9, "🧡": 0.8, "💛": 0.8, "💚": 0.7, "💙": 0.7,
    "💜": 0.7, "🖤": 0.0, "🤍": 0.1, "🤎": 0.1, "💔": -0.7,
    "❣️": 0.8, "💕": 0.8, "💞": 0.8, "💓": 0.7, "💗": 0.8,
    "💖": 0.8, "💘": 0.7, "💝": 0.8, "👍": 0.5, "👎": -0.5,
    "👏": 0.6, "🙌": 0.7, "🤝": 0.3, "💪": 0.5, "🔥": 0.5,
    "⭐": 0.5, "🌟": 0.5, "💯": 0.6, "✅": 0.4, "❌": -0.4,
    "⚠️": -0.1, "🎉": 0.7, "🎊": 0.7, "🏆": 0.8, "🥇": 0.8,
    "💀": -0.2, "👻": 0.1, "🤡": -0.3, "💩": -0.4, "🙈": 0.3,
    "🙉": 0.1, "🙊": 0.2, "😢": -0.6, "😡": -0.8,
}


def extract_emoji_sentiment(text):
    """Extract emoji-based sentiment from text.

    Returns dict: {emoji_sentiment, emoji_count, emoji_list}.
    Based on Unicode CLDR emoji conventions.
    """
    emojis = []
    for char in text:
        if char in EMOJI_SENTIMENT:
            emojis.append({"emoji": char, "polarity": EMOJI_SENTIMENT[char]})
    if not emojis:
        return {"emoji_sentiment": 0.0, "emoji_count": 0, "emoji_list": []}
    mean_pol = float(np.mean([e["polarity"] for e in emojis]))
    return {"emoji_sentiment": round(mean_pol, 4), "emoji_count": len(emojis),
            "emoji_list": emojis}


# ─── S11: Sentiment Explanation ─────────────────────────────────────────────

def explain_sentiment(text, lex_scores=None, transformer_results=None):
    """Generate human-readable explanation of why text was classified as positive/negative.

    Uses keyword-based + lexicon score attribution.
    """
    explanation = []

    # High-impact negative words
    NEGATIVE_KEYWORDS = {
        "terrible": -0.9, "horrible": -0.85, "awful": -0.85, "worst": -0.9,
        "dirty": -0.7, "rude": -0.75, "disgusting": -0.85, "hate": -0.8,
        "broken": -0.6, "disappointed": -0.65, "angry": -0.7, "furious": -0.8,
        "waste": -0.5, "useless": -0.6, "pathetic": -0.75, "nightmare": -0.7,
        "scam": -0.8, "fraud": -0.85, "avoid": -0.6, "never": -0.3,
    }
    POSITIVE_KEYWORDS = {
        "amazing": 0.9, "excellent": 0.85, "wonderful": 0.85, "perfect": 0.9,
        "best": 0.8, "love": 0.85, "great": 0.75, "fantastic": 0.85,
        "friendly": 0.65, "helpful": 0.65, "clean": 0.5, "comfortable": 0.6,
        "recommend": 0.7, "beautiful": 0.75, "delicious": 0.7, "outstanding": 0.85,
    }

    words = re.findall(r"[a-zA-Z']+", text.lower())
    neg_hits = []
    pos_hits = []
    for w in words:
        if w in NEGATIVE_KEYWORDS:
            neg_hits.append((w, NEGATIVE_KEYWORDS[w]))
        if w in POSITIVE_KEYWORDS:
            pos_hits.append((w, POSITIVE_KEYWORDS[w]))

    neg_hits.sort(key=lambda x: x[1])
    pos_hits.sort(key=lambda x: x[1], reverse=True)

    if neg_hits:
        explanation.append("Negative indicators: " + ", ".join(f'"{w}" ({s:+.2f})' for w, s in neg_hits[:5]))
    if pos_hits:
        explanation.append("Positive indicators: " + ", ".join(f'"{w}" ({s:+.2f})' for w, s in pos_hits[:5]))

    if lex_scores:
        top_lex = max(lex_scores.items(), key=lambda x: abs(x[1]))
        explanation.append(f"Strongest lexicon signal: {top_lex[0]} ({top_lex[1]:+.4f})")

    if transformer_results and transformer_results.get("sentiment"):
        tr = transformer_results["sentiment"]
        explanation.append(f"Transformer ({tr.get('model', 'unknown')}: {tr.get('confidence', 0):.1%} confidence)")

    if not explanation:
        explanation.append("No strong keyword signals — decision based on statistical patterns across all words")

    return "; ".join(explanation)


# ─── S2: Sentiment Shift Detection ──────────────────────────────────────────

def detect_sentiment_shifts(text, window=3, threshold=0.3):
    """Detect polarity shifts within a document.

    Splits text into sentences, computes per-sentence polarity, flags shifts
    where polarity changes sign or magnitude beyond threshold.
    Based on Polanyi & Zaenen (2006).
    """
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        try:
            nltk.data.find("sentiment/vader_lexicon")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
    except Exception:
        return {"has_shift": False, "shifts": [], "segments": []}

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < 2:
        return {"has_shift": False, "shifts": [], "segments": []}

    segments = []
    for s in sentences:
        score = sia.polarity_scores(s)["compound"]
        segments.append({"text": s, "polarity": round(score, 4),
                         "label": "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"})

    shifts = []
    for i in range(1, len(segments)):
        diff = abs(segments[i]["polarity"] - segments[i - 1]["polarity"])
        if diff >= threshold:
            shifts.append({
                "position": i,
                "from_label": segments[i - 1]["label"],
                "to_label": segments[i]["label"],
                "from_polarity": segments[i - 1]["polarity"],
                "to_polarity": segments[i]["polarity"],
                "magnitude": round(diff, 4),
            })

    return {
        "has_shift": len(shifts) > 0,
        "shift_count": len(shifts),
        "segments": segments,
        "shifts": shifts,
        "overall_arc": f"{segments[0]['label']} → {segments[-1]['label']}" if segments else "",
    }


# ─── S19: Sentiment Timeline ────────────────────────────────────────────────

def generate_sentiment_timeline(df, date_col=None, text_col="clean_text",
                                output_dir=None, group_col=None):
    """Plot sentiment polarity over time if date column available.

    Uses rolling average with configurable window. Interactive Plotly chart.
    """
    if date_col is None:
        for c in ["date", "timestamp", "created_at", "time", "datetime", "year"]:
            if c in df.columns:
                date_col = c
                break
    if date_col is None or date_col not in df.columns:
        return None

    df = df.copy()
    try:
        df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
    except Exception:
        return None
    df = df.dropna(subset=["_date"]).sort_values("_date")
    if len(df) < 3:
        return None

    sia = None
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        try:
            nltk.data.find("sentiment/vader_lexicon")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
    except Exception:
        pass
    if sia is None:
        return None

    df["_polarity"] = df[text_col].apply(lambda x: sia.polarity_scores(str(x))["compound"])
    agg = df.groupby("_date")["_polarity"].mean().reset_index()
    agg.columns = ["date", "polarity"]

    if group_col and group_col in df.columns:
        groups = df[group_col].unique()
        agg_groups = df.groupby(["_date", group_col])["_polarity"].mean().reset_index()
    else:
        groups = []
        agg_groups = None

    out_json = None
    out_path = os.path.join(output_dir, "sentiment_timeline.json") if output_dir else None

    if HAS_PLOTLY:
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            window = min(5, len(agg) // 2) if len(agg) > 4 else 1
            if window < 1:
                window = 1
            agg["rolling"] = agg["polarity"].rolling(window, center=True).mean()
            fig.add_trace(go.Scatter(
                x=agg["date"], y=agg["polarity"], mode="markers",
                marker=dict(size=6, opacity=0.4, color="#3498db"),
                name="Raw polarity"))
            fig.add_trace(go.Scatter(
                x=agg["date"], y=agg["rolling"], mode="lines",
                line=dict(width=3, color="#e74c3c"),
                name=f"Rolling avg ({window})"))

            if agg_groups is not None:
                colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
                for i, grp in enumerate(groups):
                    gdata = agg_groups[agg_groups[group_col] == grp]
                    gdata = gdata.sort_values("_date")
                    gr = gdata["_polarity"].rolling(window, center=True).mean()
                    fig.add_trace(go.Scatter(
                        x=gdata["_date"], y=gr, mode="lines",
                        line=dict(width=2, color=colors[i % len(colors)]),
                        name=str(grp)))

            fig.update_layout(title="Sentiment Timeline", xaxis_title="Date",
                              yaxis_title="Polarity", template="plotly_white",
                              height=400)
            if out_path:
                fig.write_html(out_path.replace(".json", ".html"))
                print(f"  Saved sentiment timeline to sentiment_timeline.html")
        except Exception:
            pass

    out_json = agg.to_dict(orient="records")
    return out_json


# ─── Sub-functions for run() ─────────────────────────────────────────────────


def _print_banner(tier, run_absa, run_embeddings):
    model_names = {"full": "RoBERTa (sentiment + emotion + GoEmotions)", "lightweight": "DistilBERT"}
    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSIS ULTRA v3")
    print("=" * 70)
    print(f"Models: {model_names.get(tier, tier)}")
    print("Calibration: MultiLexScaled z-score")
    print("Lexicons: VADER, AFINN, NRC VAD, SentiWordNet")
    print("Emotion: NRC EmoLex + RoBERTa (11 emotions)")
    print("ABSA: PyABSA" if run_absa else "ABSA: --absa to enable")
    print("Embeddings: SBERT semantic clusters" if run_embeddings else "Embeddings: disabled")
    print()


def _run_document_analysis(ensemble, df, max_docs, quiet, group_col=None, run_absa=False, run_embeddings=True):
    print("--- Deep-Dive ---")
    test = "I feel so hopeless and alone, but people keep telling me it will get better."
    r = ensemble.analyze_text(test, doc_id="demo", calibrate=False)
    print(f"  Text: {test}")
    print(f"  Final: {r['ensemble_polarity']:+.4f} ({r['ensemble_label']})")
    tr = r.get("transformer", {})
    if tr.get("available"):
        print(f"  Sentiment: {tr['transformer_polarity']:+.4f} ({tr['transformer_label']})")
        emo = tr.get("results", {}).get("emotion", {})
        if emo: print(f"  Emotion: {emo.get('top_emotion')} ({emo.get('confidence'):.1%})")
        ge = tr.get("results", {}).get("goemotions", {})
        if ge: print(f"  GoEmotions: {ge.get('top_emotion')} ({ge.get('confidence'):.1%})")
    print(f"  Lexicons: VADER={r['lexicon_raw']['vader']:+.4f}  AFINN={r['lexicon_raw']['afinn']:+.4f}  "
          f"NRC_VAD={r['lexicon_raw']['nrc_vad']:+.4f}  SWN={r['lexicon_raw']['swn']:+.4f}")
    if any(v > 0 for v in r.get("nrc_emotions", {}).values()):
        top = max(r["nrc_emotions"], key=r["nrc_emotions"].get)
        print(f"  NRC EmoLex: {top} ({r['nrc_emotions'][top]:.3f})")
    if r.get("roberta_emotions"):
        top = max(r["roberta_emotions"], key=r["roberta_emotions"].get)
        print(f"  RoBERTa emotion: {top} ({r['roberta_emotions'][top]:.1%})")

    print("\n--- Document Analysis ---")
    text_col = "text" if "text" in df.columns else "clean_text"
    doc_result = ensemble.analyze_documents(
        df,
        text_col=text_col,
        group_col=group_col,
        max_docs=min(len(df), max_docs),
        run_absa=run_absa,
        run_embeddings=run_embeddings,
    )

    if not quiet:
        for doc in doc_result["documents"]:
            pol = doc["ensemble_polarity"]
            label = doc["ensemble_label"].upper()[:3]
            preview = doc["text_preview"][:50]
            print(f"  [{label}] {pol:+.4f}  {preview}")

    s = doc_result["summary"]
    print(f"\n--- Summary ---")
    print(f"  Documents: {s['doc_count']}")
    print(f"  Mean: {s['mean_polarity']:+.4f}  Median: {s['median_polarity']:+.4f}  Std: {s['std_polarity']:.4f}")
    print(f"  Labels: {s['label_distribution']}")
    print(f"\n--- Calibration ---")
    for lex, st in s.get("calibration_stats", {}).items():
        print(f"  {lex}: mean={st['mean']:+.4f}  std={st['std']:.4f}")

    if doc_result["group_summary"]:
        print(f"\n--- Group Comparison ---")
        for g, gs in doc_result["group_summary"].items():
            if isinstance(gs, dict) and "count" in gs:
                print(f"  {g}: n={gs['count']}  mean={gs['mean_polarity']:+.4f}  std={gs['std_polarity']:.4f}")
        if "effect_size" in doc_result["group_summary"]:
            es = doc_result["group_summary"]["effect_size"]
            print(f"  Cohen's d: {es['cohens_d']:+.4f} ({es['interpretation']})")

    sem = doc_result.get("semantic_profile", {})
    if sem.get("available"):
        print(f"\n--- Semantic Clusters ---")
        print(f"  Model: {sem.get('embedding_model')}  clusters={sem.get('n_clusters')}")
        for cl in sem.get("clusters", []):
            print(f"  C{cl['cluster']}: n={cl['size']} mean={cl['mean_polarity']:+.4f} labels={cl['label_distribution']}")
            print(f"      exemplar: {cl['exemplar_preview'][:120]}")
        print("  Outliers:")
        for out in sem.get("semantic_outliers", [])[:5]:
            print(f"    {out['doc_id']} C{out['cluster']} dist={out['distance']:.3f} pol={out['polarity']:+.3f}: {out['preview'][:100]}")

    if doc_result.get("absa_results"):
        print(f"\n--- ABSA ---")
        for i, ar in enumerate(doc_result["absa_results"][:5]):
            if ar.get("aspects"):
                print(f"  Doc {i}: {[(a['aspect'], a['sentiment']) for a in ar['aspects'][:5]]}")
        total = sum(len(ar.get("aspects", [])) for ar in doc_result["absa_results"])
        docs_w = sum(1 for ar in doc_result["absa_results"] if ar.get("aspects"))
        print(f"  Total: {total} aspects in {docs_w} docs")

    has_raw_body = "body" in df.columns
    if has_raw_body:
        doc_result["_raw_body"] = df["body"].fillna("").astype(str).tolist()
    print("\n--- Emoji Analysis ---")
    emoji_stats = {"total_emojis": 0, "docs_with_emojis": 0, "emoji_sentiment_mean": 0}
    emoji_pols = []
    for i, doc in enumerate(doc_result["documents"]):
        if has_raw_body and i < len(doc_result["_raw_body"]):
            emoji_text = doc_result["_raw_body"][i]
        else:
            emoji_text = doc.get("text_preview", "")
        er = extract_emoji_sentiment(emoji_text)
        doc["emoji_sentiment"] = er
        emoji_stats["total_emojis"] += er["emoji_count"]
        if er["emoji_count"] > 0:
            emoji_stats["docs_with_emojis"] += 1
            emoji_pols.append(er["emoji_sentiment"])
    if emoji_pols:
        emoji_stats["emoji_sentiment_mean"] = round(float(np.mean(emoji_pols)), 4)
    print(f"  Total emojis found: {emoji_stats['total_emojis']}")
    print(f"  Docs with emojis: {emoji_stats['docs_with_emojis']}/{len(doc_result['documents'])}")
    if emoji_pols:
        print(f"  Mean emoji polarity: {emoji_stats['emoji_sentiment_mean']:+.4f}")
    doc_result["emoji_stats"] = emoji_stats

    return doc_result


def _compute_group_comparison(group_col, doc_result, output):
    if not group_col or not doc_result.get("documents"):
        return None
    print("\n--- Group Comparison Statistics ---")
    groups_pols = {}
    for doc in doc_result["documents"]:
        g = doc.get("group", "unknown")
        groups_pols.setdefault(g, []).append(doc["ensemble_polarity"])
    group_keys = sorted(groups_pols.keys())
    if len(group_keys) == 2:
        g1, g2 = group_keys[0], group_keys[1]
        pols_g1 = np.array(groups_pols[g1])
        pols_g2 = np.array(groups_pols[g2])
        try:
            from scipy.stats import mannwhitneyu
            stat, p_val = mannwhitneyu(pols_g1, pols_g2, alternative="two-sided")
        except ImportError:
            def _rank_data(x):
                sorted_idx = np.argsort(x)
                ranks = np.empty_like(sorted_idx, dtype=float)
                ranks[sorted_idx] = np.arange(1, len(x) + 1, dtype=float)
                return ranks
            r1 = _rank_data(pols_g1)
            r2 = _rank_data(pols_g2)
            n1, n2 = len(pols_g1), len(pols_g2)
            u1 = n1 * n2 + n1 * (n1 + 1) / 2 - np.sum(r1)
            u2 = n1 * n2 + n2 * (n2 + 1) / 2 - np.sum(r2)
            stat = min(u1, u2)
            mu_u = n1 * n2 / 2
            sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            z = (stat - mu_u) / sigma_u if sigma_u > 0 else 0
            from scipy.stats import norm as _norm
            p_val = 2 * (1 - _norm.cdf(abs(z))) if sigma_u > 0 else 1.0
        pooled_std = np.sqrt((np.var(pols_g1, ddof=1) + np.var(pols_g2, ddof=1)) / 2)
        cohens_d = (np.mean(pols_g1) - np.mean(pols_g2)) / pooled_std if pooled_std > 0 else 0
        print(f"  Group '{g1}': n={len(pols_g1)}, mean={np.mean(pols_g1):+.4f}, std={np.std(pols_g1, ddof=1):.4f}")
        print(f"  Group '{g2}': n={len(pols_g2)}, mean={np.mean(pols_g2):+.4f}, std={np.std(pols_g2, ddof=1):.4f}")
        print(f"  Mann-Whitney U = {stat:.1f}, p = {p_val:.4f}")
        print(f"  Cohen's d = {cohens_d:+.4f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")
        group_comparison = {
            "g1": g1, "g2": g2,
            "g1_n": int(len(pols_g1)), "g2_n": int(len(pols_g2)),
            "g1_mean": round(float(np.mean(pols_g1)), 4), "g2_mean": round(float(np.mean(pols_g2)), 4),
            "g1_std": round(float(np.std(pols_g1, ddof=1)), 4), "g2_std": round(float(np.std(pols_g2, ddof=1)), 4),
            "mannwhitney_u": round(float(stat), 4), "p_value": round(float(p_val), 6),
            "cohens_d": round(float(cohens_d), 4),
            "effect_size": "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small",
            "significant": p_val < 0.05,
        }
        try:
            gc_df = pd.DataFrame([group_comparison])
            gc_df.to_csv(output / "group_comparison.csv", index=False)
            print(f"  Saved: {output / 'group_comparison.csv'}")
        except Exception as e:
            print(f"  [WARN] Could not save group_comparison.csv: {e}")
        return group_comparison
    else:
        print(f"  {len(group_keys)} groups found — Mann-Whitney requires exactly 2 for pairwise comparison")
        return {"groups": group_keys, "note": "more than 2 groups, pairwise comparison skipped"}


def _generate_explanations(doc_result):
    print("\n--- Sentiment Explanations (top 5 most polar) ---")
    sorted_docs = sorted(doc_result["documents"], key=lambda d: abs(d["ensemble_polarity"]), reverse=True)
    explanations = []
    for doc in sorted_docs[:5]:
        exp = explain_sentiment(
            doc.get("text_preview", ""),
            lex_scores=doc.get("lexicon_raw"),
            transformer_results=doc.get("transformer", {}).get("results"),
        )
        explanations.append({"doc_id": doc["doc_id"], "explanation": exp})
        label = doc["ensemble_label"].upper()[:3]
        print(f"  [{label}] {doc['ensemble_polarity']:+.4f}: {exp[:100]}...")
    return explanations


def _detect_sentiment_shifts(doc_result):
    print("\n--- Sentiment Shift Detection ---")
    shift_docs = [d for d in doc_result["documents"]
                  if len(d.get("text_preview", "").split(".")) >= 3]
    if shift_docs:
        n_shifts = 0
        for doc in shift_docs[:50]:
            shift = detect_sentiment_shifts(doc["text_preview"])
            doc["sentiment_shift"] = shift
            if shift["has_shift"]:
                n_shifts += 1
                print(f"  Shift in {doc['doc_id']}: {shift['overall_arc']} ({shift['shift_count']} shifts)")
        print(f"  {n_shifts}/{min(len(shift_docs), 50)} docs show polarity shifts")
    else:
        print("  No docs long enough for shift detection")


def _save_sentiment_outputs(doc_result, output):
    out_file = output / "sentiscape_ultra_results.json"
    out_file.write_text(json.dumps(doc_result, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_file}")

    csv_rows = []
    for doc in doc_result["documents"]:
        emo = doc.get("emoji_sentiment", {})
        exp = next((e["explanation"] for e in doc_result.get("explanations", []) if e["doc_id"] == doc["doc_id"]), "")
        csv_rows.append({
            "doc_id": doc["doc_id"],
            "text_preview": doc["text_preview"],
            "ensemble_polarity": doc["ensemble_polarity"],
            "ensemble_label": doc["ensemble_label"],
            "transformer_available": doc.get("transformer", {}).get("available", False),
            "nrc_top_emotion": max(doc.get("nrc_emotions", {}), key=doc.get("nrc_emotions", {}).get) if doc.get("nrc_emotions") else "",
            "emoji_sentiment": emo.get("emoji_sentiment", 0.0) if isinstance(emo, dict) else 0.0,
            "explanation": exp,
        })
    csv_df = pd.DataFrame(csv_rows)
    csv_path = output / "sentiment_by_doc.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def _generate_html_report(doc_result, output):
    s = doc_result["summary"]
    sorted_all = sorted(doc_result["documents"], key=lambda d: d["ensemble_polarity"], reverse=True)
    top10_pos = sorted_all[:10]
    top10_neg = sorted_all[-10:][::-1]
    gc = doc_result.get("group_comparison", {})
    label_dist = s.get("label_distribution", {})
    label_rows_html = "".join(
        f"<tr><td>{lbl}</td><td>{cnt}</td><td>{cnt/max(s['doc_count'],1)*100:.1f}%</td></tr>"
        for lbl, cnt in label_dist.items()
    )
    pos_rows_html = "".join(
        f"<tr><td>{d['doc_id']}</td><td>{d['ensemble_polarity']:+.4f}</td>"
        f"<td>{d['ensemble_label']}</td><td>{d['text_preview'][:80]}</td></tr>"
        for d in top10_pos
    )
    neg_rows_html = "".join(
        f"<tr><td>{d['doc_id']}</td><td>{d['ensemble_polarity']:+.4f}</td>"
        f"<td>{d['ensemble_label']}</td><td>{d['text_preview'][:80]}</td></tr>"
        for d in top10_neg
    )
    gc_html = ""
    if gc and "g1" in gc:
        gc_html = f"""
    <h2>Group Comparison</h2>
    <table>
      <tr><th>Group</th><th>n</th><th>Mean</th><th>Std</th></tr>
      <tr><td>{gc['g1']}</td><td>{gc['g1_n']}</td><td>{gc['g1_mean']:+.4f}</td><td>{gc['g1_std']:.4f}</td></tr>
      <tr><td>{gc['g2']}</td><td>{gc['g2_n']}</td><td>{gc['g2_mean']:+.4f}</td><td>{gc['g2_std']:.4f}</td></tr>
    </table>
    <p>Mann-Whitney U = {gc['mannwhitney_u']:.1f}, p = {gc['p_value']:.4f},
       Cohen's d = {gc['cohens_d']:+.4f} ({gc['effect_size']})</p>"""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><title>Sentiment Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  h1, h2 {{ color: #1a1a2e; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  .pos {{ color: #27ae60; }} .neg {{ color: #c0392b; }} .neu {{ color: #7f8c8d; }}
</style>
</head>
<body>
<h1>Sentiment Analysis Report</h1>
<p>Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>

<h2>Summary Statistics</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Documents</td><td>{s['doc_count']}</td></tr>
  <tr><td>Mean polarity</td><td>{s['mean_polarity']:+.4f}</td></tr>
  <tr><td>Median polarity</td><td>{s['median_polarity']:+.4f}</td></tr>
  <tr><td>Std polarity</td><td>{s['std_polarity']:.4f}</td></tr>
</table>

<h2>Label Distribution</h2>
<table>
  <tr><th>Label</th><th>Count</th><th>%</th></tr>
  {label_rows_html}
</table>

<h2>Top 10 Most Positive</h2>
<table>
  <tr><th>doc_id</th><th>polarity</th><th>label</th><th>preview</th></tr>
  {pos_rows_html}
</table>

<h2>Top 10 Most Negative</h2>
<table>
  <tr><th>doc_id</th><th>polarity</th><th>label</th><th>preview</th></tr>
  {neg_rows_html}
</table>

{gc_html}

<h2>Output Files</h2>
<ul>
  <li><a href="sentiscape_ultra_results.json">sentiscape_ultra_results.json</a></li>
  <li><a href="sentiment_by_doc.csv">sentiment_by_doc.csv</a></li>
  <li><a href="manifest.json">manifest.json</a></li>
  <li><a href="group_comparison.csv">group_comparison.csv</a></li>
</ul>
</body></html>"""
    report_path = output / "sentiment_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"Saved: {report_path}")

    # ── ReportBuilder: comprehensive sentiment analysis report ──
    if HAS_REPORT:
        try:
            import plotly.express as px
            docs = doc_result.get("documents", [])
            summary = doc_result.get("summary", {})
            n = len(docs)
            rb = ReportBuilder(
                "Sentiment Analysis ULTRA",
                dataset=doc_result.get("_input_file", ""),
                n_docs=n,
                extra_header=f"Profile: {doc_result.get('tier', 'full')}",
            )

            # --- KEY FINDINGS (5+) ---
            findings = []
            dist = summary.get("label_distribution", {})
            total = sum(dist.values()) or 1
            for label in ["positive", "negative", "neutral"]:
                pct = dist.get(label, 0) / total * 100
                findings.append(f"{pct:.0f}% {label} ({dist.get(label, 0)} docs)")
            mean_pol = summary.get("mean_polarity", 0)
            std_pol = summary.get("std_polarity", 0)
            findings.append(f"Mean polarity: {mean_pol:+.3f} (std={std_pol:.3f}, range: -1 to +1)")
            median_pol = summary.get("median_polarity", 0)
            findings.append(f"Median polarity: {median_pol:+.3f}")
            gc = doc_result.get("group_comparison", {})
            if gc and gc.get("significant"):
                findings.append(f"Significant group difference: d={gc.get('cohens_d',0):.2f} ({gc.get('effect_size','')})")
            if docs:
                most_pos = max(docs, key=lambda d: d.get("ensemble_polarity", 0))
                most_neg = min(docs, key=lambda d: d.get("ensemble_polarity", 0))
                findings.append(f"Most positive: '{str(most_pos.get('text_preview',''))[:80]}...' ({most_pos.get('ensemble_polarity',0):+.3f})")
                findings.append(f"Most negative: '{str(most_neg.get('text_preview',''))[:80]}...' ({most_neg.get('ensemble_polarity',0):+.3f})")
            rb.add_key_findings(findings[:7])

            # --- RATIONALE ---
            cal = summary.get("calibration_stats", {})
            cal_summary = ", ".join(f"{k}: mean={v.get('mean',0):.3f}" for k, v in cal.items()) if cal else "N/A"
            rb.add_rationale("Analysis Pipeline",
                f"Profile: {doc_result.get('tier','full')}. Ensemble: RoBERTa x 0.7 + calibrated lexicons x 0.3. "
                f"Lexicons: VADER, AFINN, NRC VAD, SentiWordNet.")
            rb.add_rationale("Calibration",
                f"MultiLexScaled z-score calibration. Calibrated lexicon mean: {summary.get('calibration_stats',{}).get('calibrated',{}).get('mean',0):.3f}")

            # --- METRICS ---
            rb.add_metric("Mean Polarity", mean_pol, thresholds=THRESHOLDS.get("polarity"))
            rb.add_metric("Std Polarity", std_pol)
            if gc and gc.get("cohens_d") is not None:
                rb.add_metric("Cohen's d (groups)", abs(gc.get("cohens_d", 0)), thresholds=THRESHOLDS.get("cohens_d"))

            # --- CHARTS ---
            if dist:
                fig_dist = px.bar(
                    x=list(dist.keys()), y=list(dist.values()),
                    labels={"x": "Sentiment", "y": "Count"},
                    title="Sentiment Distribution",
                    color=list(dist.keys()),
                    color_discrete_map={"positive": "#10b981", "negative": "#ef4444", "neutral": "#94a3b8"}
                )
                fig_dist.update_layout(showlegend=False)
                rb.add_chart(fig_dist, title="Sentiment Distribution")

            if docs:
                polarities = [d.get("ensemble_polarity", 0) for d in docs]
                fig_hist = px.histogram(
                    x=polarities, nbins=30,
                    labels={"x": "Polarity", "y": "Count"},
                    title="Polarity Distribution",
                    color_discrete_sequence=["#3b82f6"]
                )
                fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
                fig_hist.update_layout(bargap=0.05)
                rb.add_chart(fig_hist, title="Polarity Histogram")

            # Emotion breakdown chart (average NRC emotions)
            if docs:
                emo_sums = {}
                n_emo = 0
                for d in docs:
                    emo = d.get("nrc_emotions", {})
                    if emo:
                        n_emo += 1
                        for k, v in emo.items():
                            emo_sums[k] = emo_sums.get(k, 0) + v
                if emo_sums and n_emo > 0:
                    emo_avg = {k: v / n_emo for k, v in sorted(emo_sums.items(), key=lambda x: -x[1])}
                    fig_emo = px.bar(
                        x=list(emo_avg.keys()), y=list(emo_avg.values()),
                        labels={"x": "Emotion", "y": "Average Score"},
                        title=f"NRC Emotion Breakdown (n={n_emo} docs with emotions)",
                        color=list(emo_avg.keys()),
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_emo.update_layout(showlegend=False)
                    rb.add_chart(fig_emo, title="NRC Emotion Breakdown")

            # Lexicon comparison chart
            if docs:
                lex_means = {}
                for lex_name in ["vader", "afinn", "nrc_vad", "swn"]:
                    vals = [d.get("lexicon_raw", {}).get(lex_name, 0) for d in docs if d.get("lexicon_raw", {}).get(lex_name) is not None]
                    if vals:
                        lex_means[lex_name.upper()] = sum(vals) / len(vals)
                if lex_means:
                    fig_lex = px.bar(
                        x=list(lex_means.keys()), y=list(lex_means.values()),
                        labels={"x": "Lexicon", "y": "Mean Score"},
                        title="Lexicon Comparison (Raw Means)",
                        color=list(lex_means.keys()),
                        color_discrete_sequence=["#6366f1", "#f59e0b", "#10b981", "#ef4444"]
                    )
                    fig_lex.update_layout(showlegend=False)
                    rb.add_chart(fig_lex, title="Lexicon Comparison")

            # --- TABLES ---
            if docs:
                rows = []
                for d in docs:
                    emo = d.get("nrc_emotions", {})
                    top_emo = max(emo, key=emo.get) if emo else ""
                    lex = d.get("lexicon_raw", {})
                    rows.append({
                        "doc_id": str(d.get("doc_id", ""))[:16],
                        "text_preview": str(d.get("text_preview", ""))[:120],
                        "label": d.get("ensemble_label", ""),
                        "polarity": round(d.get("ensemble_polarity", 0), 4),
                        "confidence": round(d.get("transformer", {}).get("results", {}).get("sentiment", {}).get("confidence", 0) or d.get("transformer", {}).get("results", {}).get("lightweight", {}).get("confidence", 0), 4),
                        "vader": round(lex.get("vader", 0), 4),
                        "afinn": round(lex.get("afinn", 0), 4),
                        "nrc_vad": round(lex.get("nrc_vad", 0), 4),
                        "swn": round(lex.get("swn", 0), 4),
                        "top_emotion": top_emo,
                    })
                senti_df = pd.DataFrame(rows)
                rb.add_table(senti_df, title="Sentiment by Document", expand_col="text_preview")

            # Calibration stats table
            cal = summary.get("calibration_stats", {})
            if cal:
                cal_rows = [{"lexicon": k, "mean": round(v.get("mean", 0), 4), "std": round(v.get("std", 0), 4)} for k, v in cal.items()]
                cal_df = pd.DataFrame(cal_rows)
                rb.add_table(cal_df, title="Lexicon Calibration Statistics")

            # --- BUILD ---
            rb.build(output / "report.html")
            rb.build_csv(output / "raw_output.csv")
        except Exception as e:
            print(f"  [!] ReportBuilder error: {e}")


# ─── Main orchestrator ───────────────────────────────────────────────────────


def run(df, output_dir=None, run_absa=False, tier="full", run_embeddings=True,
        max_docs=500, quiet=False, group_col_override=None):
    output = Path(output_dir or "output")
    output.mkdir(exist_ok=True)
    _run_start = time.time()

    _print_banner(tier, run_absa, run_embeddings)
    ensemble = SentimentEnsemble(tier=tier)

    group_col = group_col_override or next((c for c in df.columns if c.startswith("meta_")), None)
    doc_result = _run_document_analysis(ensemble, df, max_docs, quiet, group_col, run_absa, run_embeddings)

    if group_col and group_col in df.columns:
        doc_result["group_comparison"] = _compute_group_comparison(group_col, doc_result, output)

    doc_result["explanations"] = _generate_explanations(doc_result)
    _detect_sentiment_shifts(doc_result)

    timeline = generate_sentiment_timeline(df, text_col="text" if "text" in df.columns else "clean_text",
                                           output_dir=str(output))
    if timeline:
        doc_result["sentiment_timeline"] = timeline

    _save_sentiment_outputs(doc_result, output)
    _generate_html_report(doc_result, output)

    # ── Schema JSONL + manifest ────────────────────────────────────────────
    if HAS_SCHEMA:
        std_docs = []
        for doc in doc_result.get("documents", []):
            std_doc = new_doc(
                doc.get("doc_id", ""),
                doc.get("text_preview", ""),
            )
            sentiment_data = {
                "polarity": doc.get("ensemble_polarity", 0),
                "label": doc.get("ensemble_label", "neutral"),
                "confidence": (
                    doc.get("transformer", {}).get("results", {}).get("lightweight", {}).get("confidence", 0)
                    or doc.get("transformer", {}).get("results", {}).get("sentiment", {}).get("confidence", 0)
                ),
                "transformer_polarity": doc.get("transformer", {}).get("transformer_polarity", 0),
                "lexicon_raw": doc.get("lexicon_raw", {}),
                "lexicon_calibrated": doc.get("lexicon_calibrated", 0),
                "nrc_emotions": doc.get("nrc_emotions", {}),
                "roberta_emotions": doc.get("roberta_emotions", {}),
            }
            add_tool_section(std_doc, "sentiment", sentiment_data)
            std_docs.append(std_doc)

        write_docs_jsonl(std_docs, output)
        elapsed = time.time() - _run_start
        m = build_manifest(
            "sentiscape_ultra", len(doc_result.get("documents", [])), elapsed,
            parameters={"profile": tier, "run_absa": run_absa, "run_embeddings": run_embeddings},
        )
        _write_manifest(m, output)
        print(f"Schema JSONL: {output / 'ultra_output.jsonl'}")
        print(f"Schema manifest: {output / 'manifest.json'}")
    else:
        manifest = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_docs": len(doc_result.get("documents", [])),
            "elapsed_sec": round(time.time() - _run_start, 2),
            "models_used": tier,
        }
        manifest_file = output / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest: {manifest_file}")

    return doc_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SENTIMENT ANALYSIS ULTRA v3 — Evidence-Based Sentiment Analysis")
    parser.add_argument("csv_path", nargs="?", default=None,
                        help="Path to CSV file (omit for demo)")
    parser.add_argument("text_col", nargs="?", default=None,
                        help="Column name containing text to analyse")
    parser.add_argument("-o", "--output", default="output_senti",
                        help="Output directory (default: output_senti)")
    parser.add_argument("--profile", choices=["fast", "full"], default="fast",
                        help="Model profile: fast=lightweight only, full=all models (default: fast)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly sample N documents before analysis")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--max-docs", type=int, default=500,
                        help="Maximum documents to analyse (default: 500)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-document [NEG]/[POS] output")
    parser.add_argument("--group", default=None,
                        help="Column name for contrastive group analysis")
    parser.add_argument("--absa", action="store_true",
                        help="Enable ABSA (aspect-based sentiment analysis)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Disable semantic embedding clusters")
    args = parser.parse_args()

    tier = "lightweight" if args.profile == "fast" else "full"
    df = _load_documents(args.csv_path, args.text_col)

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled {args.sample} docs (seed={args.seed})")

    run(df, output_dir=args.output, run_absa=args.absa, tier=tier,
        run_embeddings=not args.no_embeddings, max_docs=args.max_docs,
        quiet=args.quiet, group_col_override=args.group)
