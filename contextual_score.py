import sys
import argparse
import datetime
import logging
from typing import List, Dict, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover
    print("Missing dependency: requests. Install with: pip install requests", file=sys.stderr)
    raise

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    print("Missing dependency: beautifulsoup4. Install with: pip install beautifulsoup4", file=sys.stderr)
    raise

# VADER from NLTK (preferred)
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False

# TextBlob fallback
try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except Exception:
    _TEXTBLOB_AVAILABLE = False

# Optional: newspaper3k for snippet extraction (best-effort)
try:
    from newspaper import Article  # type: ignore
    _NEWSPAPER_AVAILABLE = True
except Exception:
    _NEWSPAPER_AVAILABLE = False

# Optional: spaCy for NER (best-effort)
_SPACY_AVAILABLE = False
try:
    import spacy  # type: ignore
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False


###############################################################################
# Utility and Setup
###############################################################################


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Best-effort: ensure console can handle unicode (e.g., currency symbols) on Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def ensure_vader() -> Optional[object]:
    """Ensure VADER lexicon is available and return analyzer if possible."""
    if not _NLTK_AVAILABLE:
        return None
    try:
        # Ensure the VADER lexicon is present
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


###############################################################################
# Fetching News via Google News RSS
###############################################################################


def build_google_news_rss_url(query: str, lang: str = "en", country: str = "US") -> str:
    # Google News RSS: https://news.google.com/rss/search?q=<query>&hl=<lang>&gl=<country>&ceid=<country>:<lang>
    from urllib.parse import quote_plus
    encoded = quote_plus(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl={lang}&gl={country}&ceid={country}:{lang}"


def fetch_rss_items(query: str, max_items: int = 20, timeout_sec: int = 10) -> List[Dict[str, str]]:
    """Fetch headlines from Google News RSS for a given query.

    Returns list of dicts: { 'title': str, 'link': str, 'published': str }
    """
    url = build_google_news_rss_url(query)
    try:
        resp = requests.get(url, timeout=timeout_sec)
        resp.raise_for_status()
    except Exception as exc:
        logging.warning("Failed to fetch RSS for query '%s': %s", query, exc)
        return []

    soup = BeautifulSoup(resp.text, 'xml')
    items = []
    for item in soup.find_all('item')[:max_items]:
        title = (item.title.text or "").strip() if item.title else ""
        link = (item.link.text or "").strip() if item.link else ""
        pub_date = (item.pubDate.text or "").strip() if item.pubDate else ""
        description = (item.description.text or "").strip() if item.description else ""
        if title:
            items.append({
                'title': title,
                'link': link,
                'published': pub_date,
                'description': description,
            })
    return items


###############################################################################
# Sentiment Analysis
###############################################################################


class SentimentService:
    """Provide sentiment classification with VADER preferred, TextBlob fallback."""

    def __init__(self) -> None:
        self.vader = ensure_vader()
        self.use_textblob = self.vader is None and _TEXTBLOB_AVAILABLE
        if not self.vader and not self.use_textblob:
            logging.warning("Neither VADER nor TextBlob available. Falling back to neutral.")

    @staticmethod
    def _label_from_compound(compound: float, neg_threshold: float = -0.3, pos_threshold: float = 0.3) -> int:
        # Nuanced thresholds
        if compound >= pos_threshold:
            return 1
        if compound <= neg_threshold:
            return -1
        return 0

    def analyze(self, text: str, neg_threshold: float = -0.3, pos_threshold: float = 0.3) -> Tuple[int, float]:
        """Return (label, score) where label in {-1,0,1}, score is signed strength.
        score is VADER compound in [-1,1] or scaled TextBlob polarity.
        """
        if not text:
            return 0, 0.0

        if self.vader is not None:
            try:
                scores = self.vader.polarity_scores(text)
                compound = float(scores.get('compound', 0.0))
                return self._label_from_compound(compound, neg_threshold, pos_threshold), compound
            except Exception:
                pass

        if self.use_textblob:
            try:
                polarity = float(TextBlob(text).sentiment.polarity)  # [-1,1]
                # Use same thresholds as VADER for consistency
                label = 1 if polarity >= pos_threshold else (-1 if polarity <= neg_threshold else 0)
                return label, polarity
            except Exception:
                pass

        # Fallback neutral
        return 0, 0.0


def score_articles(
    articles: List[Dict[str, str]],
    sentiment: SentimentService,
    *,
    neg_threshold: float = -0.3,
    pos_threshold: float = 0.3,
    headline_weight: float = 0.7,
    snippet_weight: float = 0.3,
    time_decay: bool = True,
    enable_ner: bool = False,
    nlp_model: Optional[object] = None,
    applicant_name: Optional[str] = None,
    sector_keywords: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Analyze list of articles and return aggregate metrics.

    Returns dict with:
      - average_label: float (mean of -1/0/1)
      - average_strength: float (mean of signed strength)
      - labeled: List[{'title','link','published','label','strength'}]
      - top_positive: top 3 by strength desc
      - top_negative: top 3 by strength asc
    """
    def parse_pub_date(pub: str) -> Optional[datetime.datetime]:
        try:
            return datetime.datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z")
        except Exception:
            try:
                return datetime.datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z")
            except Exception:
                return None

    def compute_decay(pub: str) -> float:
        if not time_decay:
            return 1.0
        dt = parse_pub_date(pub)
        if not dt:
            return 1.0
        days = (datetime.datetime.utcnow() - dt.replace(tzinfo=None)).days
        if days <= 7:
            return 1.0
        if days <= 30:
            return 0.5
        return 0.2

    def ner_person_match(text: str) -> bool:
        if not (enable_ner and nlp_model and applicant_name):
            return True
        try:
            doc = nlp_model(text)
            for ent in getattr(doc, 'ents', []):
                if ent.label_ == 'PERSON' and applicant_name.lower() in ent.text.lower():
                    return True
            return False
        except Exception:
            return True

    def sector_cooccurs(text: str) -> bool:
        if not sector_keywords:
            return True
        txt = text.lower()
        return any(kw.lower() in txt for kw in sector_keywords)

    labeled: List[Dict[str, object]] = []
    for a in articles:
        title = a.get('title', '')
        desc = a.get('description', '')
        link = a.get('link', '')
        published = a.get('published', '')

        # Optional snippet via newspaper3k when description is empty
        snippet_text = desc
        if not snippet_text and _NEWSPAPER_AVAILABLE and link:
            try:
                art = Article(link)
                art.download()
                art.parse()
                snippet_text = " ".join((art.text or "").splitlines())[:400]
            except Exception:
                snippet_text = ""

        # NER & sector co-occurrence filters
        merged_text_for_filter = f"{title}. {snippet_text}".strip()
        if not ner_person_match(merged_text_for_filter):
            continue
        if not sector_cooccurs(merged_text_for_filter):
            continue

        # Sentiments
        h_label, h_strength = sentiment.analyze(title, neg_threshold, pos_threshold)
        s_label, s_strength = sentiment.analyze(snippet_text, neg_threshold, pos_threshold) if snippet_text else (0, 0.0)

        # Blend headline and snippet strengths; if snippet missing, rely on headline
        combined_strength = (headline_weight * h_strength) + (snippet_weight * s_strength)
        combined_label = sentiment._label_from_compound(combined_strength, neg_threshold, pos_threshold)

        decay = compute_decay(published)

        labeled.append({
            'title': title,
            'link': link,
            'published': published,
            'label': combined_label,
            'strength': combined_strength,
            'headline_strength': h_strength,
            'snippet_strength': s_strength,
            'weight': decay,
        })

    if not labeled:
        return {
            'average_label': 0.0,
            'average_strength': 0.0,
            'labeled': [],
            'top_positive': [],
            'top_negative': [],
        }

    total_weight = sum(x['weight'] for x in labeled) or 1.0
    avg_label = sum(x['label'] * x['weight'] for x in labeled) / float(total_weight)
    avg_strength = sum(x['strength'] * x['weight'] for x in labeled) / float(total_weight)
    positives = sorted([x for x in labeled if x['label'] == 1], key=lambda x: x['strength'], reverse=True)[:3]
    negatives = sorted([x for x in labeled if x['label'] == -1], key=lambda x: x['strength'])[:3]
    return {
        'average_label': avg_label,
        'average_strength': avg_strength,
        'labeled': labeled,
        'top_positive': positives,
        'top_negative': negatives,
        'count': len(labeled),
    }


###############################################################################
# Scoring and Explainability
###############################################################################


def weighted_score(applicant_avg: float, sector_avg: float, market_avg: float, weights: Tuple[float, float, float]) -> int:
    """Compute weighted score in range [0,100].

    Inputs are averages in [-1,1] (mean of labels). Weights: 0.5, 0.3, 0.2.
    """
    aw, sw, mw = weights
    raw = (aw * applicant_avg) + (sw * sector_avg) + (mw * market_avg)
    normalized = int(round(((raw + 1.0) / 2.0) * 100))
    return max(0, min(100, normalized))


def sector_tilt_text(sector_avg_label: float) -> str:
    if sector_avg_label >= 0.05:
        return "Sector trend tilt: Positive"
    if sector_avg_label <= -0.05:
        return "Sector trend tilt: Negative"
    return "Sector trend tilt: Neutral"


def compute_confidence(*counts: int) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    # Simple saturation function: n/(n+5)
    conf = total / float(total + 5)
    return max(0.0, min(1.0, conf))


###############################################################################
# CLI and Orchestration
###############################################################################


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Context-Aware Credit Suitability Scoring")
    parser.add_argument("--name", type=str, help="Applicant full name")
    parser.add_argument("--occupation", type=str, help="Applicant occupation")
    parser.add_argument("--sector", type=str, help="Applicant sector/industry")
    # Additional disambiguation fields
    parser.add_argument("--age", type=str, help="Applicant age (string to allow ranges)")
    parser.add_argument("--address", type=str, help="Applicant address or city/state/country")
    parser.add_argument("--company", type=str, help="Employer/company (if applicable)")
    parser.add_argument("--lang", type=str, default="en", help="News language (default: en)")
    parser.add_argument("--country", type=str, default="US", help="News country code (default: US)")
    parser.add_argument("--max_items", type=int, default=20, help="Max headlines per query (default: 20)")
    parser.add_argument("--config", type=str, help="Path to JSON config for weights/thresholds")
    parser.add_argument("--review", action="store_true", help="Review mode: show detailed items before final score")
    parser.add_argument("--enable-ner", action="store_true", help="Enable spaCy NER filtering if available")
    return parser.parse_args(argv)


def prompt_missing(value: Optional[str], prompt_text: str) -> str:
    return value if value else input(prompt_text).strip()


def build_queries(name: str, occupation: str, sector: str, age: Optional[str], address: Optional[str], company: Optional[str]) -> Dict[str, str]:
    # Build more specific queries to reduce name collisions
    disambig_bits: List[str] = []
    if company:
        disambig_bits.append(company)
    if address:
        disambig_bits.append(address)
    if age:
        disambig_bits.append(f"age {age}")

    disambig = " ".join(b for b in disambig_bits if b)

    applicant_q = f"\"{name}\" {disambig}".strip()
    sector_q = f"{occupation} {sector} {company or ''} {address or ''}".strip()
    market_q = "economy OR market trends"
    return {
        'applicant': applicant_q,
        'sector': sector_q,
        'market': market_q,
    }


def run(name: str, occupation: str, sector: str, lang: str, country: str, max_items: int, age: Optional[str], address: Optional[str], company: Optional[str], *, config: Optional[Dict[str, object]] = None, review: bool = False, enable_ner: bool = False) -> None:
    sentiment = SentimentService()

    queries = build_queries(name, occupation, sector, age=age, address=address, company=company)
    applicant_items = fetch_rss_items(queries['applicant'], max_items=max_items)
    sector_items = fetch_rss_items(queries['sector'], max_items=max_items)
    market_items = fetch_rss_items(queries['market'], max_items=max_items)

    # Defaults
    neg_threshold = float(config.get('neg_threshold', -0.3)) if config else -0.3
    pos_threshold = float(config.get('pos_threshold', 0.3)) if config else 0.3
    headline_weight = float(config.get('headline_weight', 0.7)) if config else 0.7
    snippet_weight = float(config.get('snippet_weight', 0.3)) if config else 0.3
    applicant_w = float(config.get('applicant_weight', 0.5)) if config else 0.5
    sector_w = float(config.get('sector_weight', 0.3)) if config else 0.3
    market_w = float(config.get('market_weight', 0.2)) if config else 0.2

    # Sector keywords for co-occurrence checks
    sector_keywords = list({
        *(occupation.split() if occupation else []),
        *(sector.split() if sector else []),
        *(company.split() if company else []),
    })

    nlp_model = None
    if enable_ner and _SPACY_AVAILABLE:
        try:
            # Try to load a small English model; skip if unavailable
            nlp_model = spacy.load("en_core_web_sm")  # type: ignore
        except Exception:
            nlp_model = None

    applicant_result = score_articles(
        applicant_items, sentiment,
        neg_threshold=neg_threshold, pos_threshold=pos_threshold,
        headline_weight=headline_weight, snippet_weight=snippet_weight,
        time_decay=True, enable_ner=enable_ner, nlp_model=nlp_model,
        applicant_name=name, sector_keywords=sector_keywords,
    )
    sector_result = score_articles(
        sector_items, sentiment,
        neg_threshold=neg_threshold, pos_threshold=pos_threshold,
        headline_weight=headline_weight, snippet_weight=snippet_weight,
        time_decay=True, enable_ner=False, nlp_model=None,
        applicant_name=None, sector_keywords=sector_keywords,
    )
    market_result = score_articles(
        market_items, sentiment,
        neg_threshold=neg_threshold, pos_threshold=pos_threshold,
        headline_weight=headline_weight, snippet_weight=snippet_weight,
        time_decay=True, enable_ner=False, nlp_model=None,
        applicant_name=None, sector_keywords=None,
    )

    final_score = weighted_score(
        applicant_result['average_label'],
        sector_result['average_label'],
        market_result['average_label'],
        (applicant_w, sector_w, market_w),
    )

    confidence = compute_confidence(
        int(applicant_result.get('count', 0)),
        int(sector_result.get('count', 0)),
        int(market_result.get('count', 0)),
    )

    # Output
    print("")
    # Compose details line
    details_parts: List[str] = [occupation, sector]
    if company:
        details_parts.append(company)
    if address:
        details_parts.append(address)
    if age:
        details_parts.append(f"Age: {age}")

    details = ", ".join([p for p in details_parts if p])
    print(f"Applicant: {name} ({details})")
    print(f"Contextual Suitability Score: {final_score} / 100")
    if (applicant_result.get('count', 0) == 0 and sector_result.get('count', 0) == 0 and market_result.get('count', 0) == 0):
        print("Note: No relevant news found — defaulting to neutral (low confidence).")
    print(f"Confidence: {confidence:.2f}")
    print("")

    print("Reasons:")
    # Top positives (from any bucket; prioritize applicant/sector)
    reasons_pos = (
        applicant_result['top_positive'][:3]
        or sector_result['top_positive'][:3]
        or market_result['top_positive'][:3]
    )
    reasons_neg = (
        applicant_result['top_negative'][:3]
        or sector_result['top_negative'][:3]
        or market_result['top_negative'][:3]
    )

    if not reasons_pos and not reasons_neg:
        print("- No salient headlines found. Using neutral baseline.")
    else:
        for r in reasons_pos[:3]:
            print(f"+ {r['title']}")
        for r in reasons_neg[:3]:
            print(f"- {r['title']}")

    print("")
    print(sector_tilt_text(sector_result['average_label']))

    # Sector tilt percentages
    sector_labeled = sector_result.get('labeled', [])
    if sector_labeled:
        pos_ct = sum(1 for x in sector_labeled if x['label'] == 1)
        neg_ct = sum(1 for x in sector_labeled if x['label'] == -1)
        tot = max(1, pos_ct + neg_ct)
        print(f"Sector tilt breakdown: Positive {int(round(100*pos_ct/tot))}%, Negative {int(round(100*neg_ct/tot))}%")

    # Review mode details
    if review:
        print("")
        print("Review Mode: Detailed Items")
        def _print_bucket(title: str, items: List[Dict[str, object]]):
            print(f"- {title} ({len(items)} items)")
            for it in items[:10]:
                print(f"  * [{it['label']:+}] {it['strength']:.2f} w={it.get('weight',1.0):.2f} :: {it['title']}")
        _print_bucket("Applicant", applicant_result.get('labeled', []))
        _print_bucket("Sector", sector_result.get('labeled', []))
        _print_bucket("Market", market_result.get('labeled', []))


def main(argv: Optional[List[str]] = None) -> None:
    configure_logging()
    args = parse_args(argv)
    name = prompt_missing(args.name, "Enter applicant name: ")
    occupation = prompt_missing(args.occupation, "Enter occupation: ")
    sector = prompt_missing(args.sector, "Enter sector/industry: ")
    age = args.age if args.age else input("Enter age (optional): ").strip()
    address = args.address if args.address else input("Enter address/city/state/country (optional): ").strip()
    company = args.company if args.company else input("Enter company/employer (optional): ").strip()
    config: Optional[Dict[str, object]] = None
    if args.config:
        try:
            import json
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as exc:
            logging.warning("Failed to load config '%s': %s", args.config, exc)

    try:
        run(
            name=name,
            occupation=occupation,
            sector=sector,
            lang=args.lang,
            country=args.country,
            max_items=args.max_items,
            age=age,
            address=address,
            company=company,
            config=config,
            review=args.review,
            enable_ner=args.enable_ner,
        )
    except KeyboardInterrupt:  # pragma: no cover
        print("\nAborted by user.")
    except Exception as exc:  # pragma: no cover
        logging.error("Unexpected error: %s", exc)
        print("An unexpected error occurred. Defaulting to neutral result.")
        # Neutral fallback output
        print("")
        details = ", ".join([p for p in [occupation, sector, (company or None), (address or None), (f"Age: {age}" if age else None)] if p])
        print(f"Applicant: {name} ({details})")
        print("Contextual Suitability Score: 50 / 100")
        print("")
        print("Reasons:")
        print("- Unable to compute due to an error. Neutral baseline applied.")


if __name__ == "__main__":
    main()


