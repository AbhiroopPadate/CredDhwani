## Context-Aware Credit Suitability Scoring (Prototype)

This repository contains a single-file Python prototype that computes a contextual credit suitability score for a loan applicant by combining recent news sentiment about:
- The specific applicant (disambiguated via extra attributes)
- The applicant’s occupation/sector
- General market/economy trends

The output is a 0–100 score with brief, explainable reasons (top positive/negative headlines) and a sector tilt.


### Key Features
- Lightweight and free: uses `requests`, `beautifulsoup4`, `nltk` (VADER), and `textblob`.
- News via Google News RSS (no API keys).
- VADER sentiment with TextBlob fallback; neutral fallback if both unavailable.
- Weighted scoring: Applicant 50%, Sector 30%, Market 20%.
- Explainability: prints top 3 positive and negative headlines and indicates sector trend.
- Disambiguation inputs to reduce name collisions: `age`, `address`, `company`.


### File Structure
- `contextual_score.py`: Single runnable script with modular functions for fetch, sentiment, scoring, and explanation.


### Requirements
- Python 3.x
- Packages (all free):
  - `requests`, `beautifulsoup4`, `nltk`, `vaderSentiment`, `textblob`, `regex`, `soupsieve`

Install dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install requests beautifulsoup4 nltk vaderSentiment textblob regex soupsieve
```


### How It Works
1. Input collection
   - CLI flags or interactive prompts capture: `name`, `occupation`, `sector`, plus optional `age`, `address`, and `company` to refine search queries and reduce false matches.

2. Data collection (free resources)
   - Builds Google News RSS queries for three buckets: Applicant, Sector, Market.
   - Fetches and parses RSS items with `requests` + `BeautifulSoup`.

3. Sentiment analysis
   - Prefer NLTK VADER. Automatically downloads the VADER lexicon if missing.
   - Fallback to TextBlob polarity if VADER is unavailable.
   - Final fallback is neutral.
   - Label mapping: Negative → -1, Neutral → 0, Positive → +1.

4. Scoring (0–100)
   - Compute per-bucket average label in [-1, 1].
   - Weighted blend: Applicant 50%, Sector 30%, Market 20%.
   - Normalize to [0, 100] via ((score + 1) / 2) × 100.

5. Explainability
   - Prints top 3 positive and top 3 negative headlines (highest magnitude first).
   - Prints a sector trend tilt based on average sector label (Positive/Neutral/Negative).

6. Output
   - Example (values illustrative):
```text
Applicant: Rajesh Kumar (Farmer, Agriculture, Self-employed, Patna, Bihar, India, Age: 42)
Contextual Suitability Score: 57 / 100

Reasons:
+ Bihar gets award for record egg production - Times of India
+ Prashant Kishor charts out five-point agenda ... - The New Indian Express
+ Spowdi collaborates with ... SEWA to scale Smart Farming - RuralVoice
- Rising fuel prices worry farmers in Bihar - Times of India
- Chouhan warns of strict action ... fertilisers - Times of India

Sector trend tilt: Positive
```


### Usage
Basic run (you’ll be prompted if flags are omitted):
```bash
python contextual_score.py --name "Rajesh Kumar" --occupation "Farmer" --sector "Agriculture"
```

Disambiguated run:
```bash
python contextual_score.py \
  --name "Rajesh Kumar" \
  --occupation "Farmer" \
  --sector "Agriculture" \
  --age 42 \
  --address "Patna, Bihar, India" \
  --company "Self-employed"
```

Available CLI flags:
- `--name` (str): Applicant full name
- `--occupation` (str): Applicant occupation
- `--sector` (str): Applicant sector/industry
- `--age` (str, optional): Age or range (free text, used for query disambiguation)
- `--address` (str, optional): City/state/country (used for query disambiguation)
- `--company` (str, optional): Employer/company name (disambiguation)
- `--lang` (str, default `en`): News language for Google News RSS
- `--country` (str, default `US`): Country code for Google News RSS
- `--max_items` (int, default `20`): Max headlines fetched per query


### Design Notes
- Query disambiguation: The script combines `name` with extra attributes (company/address/age) to reduce collisions with people of the same name.
- Applicant/sector/market buckets are scored separately, then combined by weights.
- The script handles missing data gracefully and defaults to neutral when needed.


### Error Handling and Fallbacks
- Network failures or empty RSS results: neutral sentiment and clear message.
- Sentiment libraries missing: neutral fallback.
- Parsing failures: safely ignored with warnings, neutral if required.


### Limitations
- RSS headlines can still include unrelated results despite disambiguation.
- Headline-only sentiment may miss context within articles.
- This is a prototype and not a production credit model.


### Privacy and Responsible Use
- Only publicly available headlines via Google News RSS are used; no scraping of personal data beyond what you provide.
- Do not use this prototype as the sole basis for lending decisions.


### Future Improvements
- Add per-country/regional news tuning and language detection.
- Integrate additional free sources (e.g., sector-specific feeds).
- Cache results and add basic deduplication.
- Add more robust entity disambiguation (NER + knowledge graphs) while staying in free-tier tooling.

# CredDhwani
AI - Powered Credit Risk Analyzer
