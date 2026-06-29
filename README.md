# NLP Labs

Covers core NLP techniques from tokenization through neural sequence labeling and LLM-based information extraction.

---

## Lab 1 — Tokenization & Zipf's Law

**Directory:** `Lab1/`

Implements a custom English tokenizer from scratch and compares it against NLTK's `word_tokenize` and `sent_tokenize`.

- Handles punctuation splitting, decimal number preservation, and case normalization
- Computes vocabulary statistics: token frequency, unique tokens, singleton (hapax legomena) rates
- Produces a log-log Zipf's Law plot of rank vs. frequency

**Key files:** `lab1.py`, `lab1.ipynb`  
**Data:** `tokens.txt`, `sentences.txt`

---

## Lab 2 — Character-Level Language Models

**Directory:** `lab2/`

Builds character-level n-gram language models and applies them to two classification tasks.

- Trains models using padded character n-gram counts (order 1–4)
- Computes per-character **perplexity** and **smoothed perplexity** (additive smoothing with ε = 1e-7)
- **Language identification:** classifies sentences across 6 European languages (EN, DE, FR, IT, DA, NL) — achieves ~99–100% accuracy at order 3+
- **Tennis commentary gender classification:** predicts whether commentary describes a male or female player — best results at bigram/trigram order (~66–72% F, ~60–65% M accuracy)

**Key files:** `charlm.py`, `lab2.ipynb`  
**Data:** `en/de/fr/it/da.train.txt`, `tennis.train.txt`, `tennis.test.txt`, `subtitles.txt`

---

## Lab 3 — Sentiment Classification

**Directory:** `lab3/`

Binary sentiment classification (positive/negative) on Yelp reviews (2000 train, dev, test splits).

- **Naive Bayes** with Bag-of-Words vectorizer (lowercase, stop word removal, min/max doc frequency filtering) — F1 ≈ 0.824
- **Linear SVM** with the same BoW features — F1 ≈ 0.797
- **Best model:** Naive Bayes + TF-IDF with unigrams and bigrams — F1 ≈ 0.838
- Includes custom precision/recall/F1 evaluation and error analysis on misclassified examples

**Key files:** `Lab3.ipynb`  
**Data:** `train.tsv`, `dev.tsv`, `test.tsv` (columns: Stars, Docid, Text)

---

## Lab 4 — (See PDF)

**Directory:** `lab4/`  
Assignment details in `Lab 4pdf.pdf`.

---

## Lab 5 — Named Entity Recognition (FLAIR LSTM-CRF)

**Directory:** `lab5/`

Trains and evaluates a neural NER tagger using the [FLAIR](https://github.com/flairNLP/flair) framework on the CoNLL 2003 English dataset.

- Architecture: Bi-LSTM + CRF with stacked GloVe + forward/backward Flair embeddings
- Entity types: PER, ORG, LOC, MISC
- Experiments: base run, data size ablation (25K-line chunks), type-collapsed training (all types → ENT), custom sentence evaluation
- Achieves F1 > 95% on the dev set within 20 epochs

**Key files:** `Lab05 Fall 2025.ipynb`  
**Data:** CoNLL 2003 English (`train.txt`, `valid.txt`, `test.txt`) — obtained separately from course Canvas

---

## Lab 6 — (See PDF)

**Directory:** `lab6/`  
Assignment details in `Lab6.pdf`.

---

## Lab 7 — WordNet & Hearst Pattern Extraction

**Directory:** `Lab7/`

Two-part lab on lexical semantics and unsupervised relation extraction.

**Part A — WordNet exploration (NLTK):**
- Synsets, hypernyms, hyponyms, troponyms, instance hyponyms
- Lowest common hypernym (e.g., dog + insect → animal)

**Part B — Hearst pattern extraction from Wikipedia:**
- Extracts hyponym/hypernym pairs using lexico-syntactic patterns ("such as", "including")
- Applied to `wiki.train.txt` and `wiki.test.txt`
- Evaluates precision of extracted relations against labeled pairs

**Key files:** `NLP_Lab_7.ipynb`

---

## Lab 8 — Information Extraction: Traditional NLP vs. LLM

**Directory:** `Lab8/`

Extracts structured fields from newspaper obituaries using two approaches, then compares results against ground truth.

**Fields extracted:** name, sex, age, locations, spouse, birthdate, occupation

**Traditional (spaCy + regex):**
- Named entity recognition for locations/organizations
- Regex patterns for age, birthdate, spouse
- Token list matching for occupations and gender pronouns
- Strong precision but low recall on complex fields (occupation F1 ≈ 30%, spouse F1 ≈ 50%)

**LLM (GPT-3.5-turbo):**
- Zero-shot JSON extraction via a single prompt
- Dramatically better recall across all fields (occupation F1 ≈ 74%, spouse F1 ≈ 100%)
- Locations over-extracted (high recall, low precision) due to listing all mentioned cities rather than place of residence

**Key files:** `NLP_Lab_8.ipynb`  
**Data:** `obits.train.txt`, `obits.test.txt` (accessed via Google Drive on Colab)

---

## Setup

Most labs run in Jupyter notebooks. Labs 5, 7, and 8 were developed on Google Colab with GPU support.

```bash
pip install nltk scikit-learn matplotlib numpy
pip install flair          # Lab 5
pip install spacy openai   # Lab 8
python -m spacy download en_core_web_sm
```
