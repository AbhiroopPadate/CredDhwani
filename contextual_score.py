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


###############################################################################
# Utility and Setup
###############################################################################


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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
        if title:
            items.append({
                'title': title,
                'link': link,
                'published': pub_date,
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
    def _label_from_compound(compound: float) -> int:
        # Standard VADER thresholds
        if compound >= 0.05:
            return 1
        if compound <= -0.05:
            return -1
        return 0

    def analyze(self, text: str) -> Tuple[int, float]:
        """Return (label, score) where label in {-1,0,1}, score is signed strength.
        score is VADER compound in [-1,1] or scaled TextBlob polarity.
        """
        if not text:
            return 0, 0.0

        if self.vader is not None:
            try:
                scores = self.vader.polarity_scores(text)
                compound = float(scores.get('compound', 0.0))
                return self._label_from_compound(compound), compound
            except Exception:
                pass

        if self.use_textblob:
            try:
                polarity = float(TextBlob(text).sentiment.polarity)  # [-1,1]
                # Use same thresholds as VADER for consistency
                label = 1 if polarity >= 0.05 else (-1 if polarity <= -0.05 else 0)
                return label, polarity
            except Exception:
                pass

        # Fallback neutral
        return 0, 0.0


def score_articles(articles: List[Dict[str, str]], sentiment: SentimentService) -> Dict[str, object]:
    """Analyze list of articles and return aggregate metrics.

    Returns dict with:
      - average_label: float (mean of -1/0/1)
      - average_strength: float (mean of signed strength)
      - labeled: List[{'title','link','published','label','strength'}]
      - top_positive: top 3 by strength desc
      - top_negative: top 3 by strength asc
    """
    labeled: List[Dict[str, object]] = []
    for a in articles:
        label, strength = sentiment.analyze(a.get('title', ''))
        labeled.append({
            'title': a.get('title', ''),
            'link': a.get('link', ''),
            'published': a.get('published', ''),
            'label': label,
            'strength': strength,
        })

    if not labeled:
        return {
            'average_label': 0.0,
            'average_strength': 0.0,
            'labeled': [],
            'top_positive': [],
            'top_negative': [],
        }

    avg_label = sum(x['label'] for x in labeled) / float(len(labeled))
    avg_strength = sum(x['strength'] for x in labeled) / float(len(labeled))
    positives = sorted([x for x in labeled if x['label'] == 1], key=lambda x: x['strength'], reverse=True)[:3]
    negatives = sorted([x for x in labeled if x['label'] == -1], key=lambda x: x['strength'])[:3]
    return {
        'average_label': avg_label,
        'average_strength': avg_strength,
        'labeled': labeled,
        'top_positive': positives,
        'top_negative': negatives,
    }


###############################################################################
# Scoring and Explainability
###############################################################################


def weighted_score(applicant_avg: float, sector_avg: float, market_avg: float) -> int:
    """Compute weighted score in range [0,100].

    Inputs are averages in [-1,1] (mean of labels). Weights: 0.5, 0.3, 0.2.
    """
    raw = (0.5 * applicant_avg) + (0.3 * sector_avg) + (0.2 * market_avg)
    normalized = int(round(((raw + 1.0) / 2.0) * 100))
    return max(0, min(100, normalized))


def sector_tilt_text(sector_avg_label: float) -> str:
    if sector_avg_label >= 0.05:
        return "Sector trend tilt: Positive"
    if sector_avg_label <= -0.05:
        return "Sector trend tilt: Negative"
    return "Sector trend tilt: Neutral"


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


def run(name: str, occupation: str, sector: str, lang: str, country: str, max_items: int, age: Optional[str], address: Optional[str], company: Optional[str]) -> None:
    sentiment = SentimentService()

    queries = build_queries(name, occupation, sector, age=age, address=address, company=company)
    applicant_items = fetch_rss_items(queries['applicant'], max_items=max_items)
    sector_items = fetch_rss_items(queries['sector'], max_items=max_items)
    market_items = fetch_rss_items(queries['market'], max_items=max_items)

    applicant_result = score_articles(applicant_items, sentiment)
    sector_result = score_articles(sector_items, sentiment)
    market_result = score_articles(market_items, sentiment)

    final_score = weighted_score(
        applicant_result['average_label'],
        sector_result['average_label'],
        market_result['average_label'],
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


def main(argv: Optional[List[str]] = None) -> None:
    configure_logging()
    args = parse_args(argv)
    name = prompt_missing(args.name, "Enter applicant name: ")
    occupation = prompt_missing(args.occupation, "Enter occupation: ")
    sector = prompt_missing(args.sector, "Enter sector/industry: ")
    age = args.age if args.age else input("Enter age (optional): ").strip()
    address = args.address if args.address else input("Enter address/city/state/country (optional): ").strip()
    company = args.company if args.company else input("Enter company/employer (optional): ").strip()

    try:
        run(name=name, occupation=occupation, sector=sector, lang=args.lang, country=args.country, max_items=args.max_items, age=age, address=address, company=company)
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


