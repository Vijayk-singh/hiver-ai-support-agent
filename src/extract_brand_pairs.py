import os
import re
import html
import argparse
from pathlib import Path
import pandas as pd
from langdetect import detect, DetectorFactory

# Set seed for reproducible language detection
DetectorFactory.seed = 42

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_CSV = PROJECT_ROOT / "data" / "raw" / "twcs.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_BRAND = "AmazonHelp"
DEFAULT_MAX_PAIRS = 20000


def clean_tweet_text(text: str) -> str:
    """Clean tweet text: unescape HTML, clean handles/punctuation, mask PII, normalize spaces."""
    if not isinstance(text, str):
        return ""
    # Unescape HTML entities (e.g. &amp; -> &, &gt; -> >)
    text = html.unescape(text)

    # Remove leading dots/spaces before handles (e.g. ".@AmazonHelp")
    text = re.sub(r'^[.\s]+(?=@)', '', text)

    # Remove all leading @handles (e.g. "@AmazonHelp @115820")
    text = re.sub(r'^(@\w+\s*)+', '', text)

    # Strip leftover leading punctuation like commas, colons, dashes left from handle separation
    text = re.sub(r'^[,\.:;\-\s]+', '', text)

    # Replace Kaggle anonymized handle for Amazon (@115821) with @Amazon
    text = re.sub(r'@115821\b', '@Amazon', text)

    # Replace other anonymized numeric user handles (e.g. @115840) with @user
    text = re.sub(r'@\d{4,}\b', '@user', text)

    # Mask Amazon order IDs (e.g., 123-1234567-1234567)
    text = re.sub(r'\b\d{3}-\d{7}-\d{7}\b', '[ORDER_ID]', text)
    # Mask emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Mask phone numbers (10-12 digits)
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{10,12}\b', '[PHONE]', text)

    # Normalize whitespace
    return ' '.join(text.split()).strip()


def is_ascii_candidate(text: str, min_ratio: float = 0.80) -> bool:
    """Fast pre-filter to eliminate non-Latin scripts (Japanese, Cyrillic, Arabic, etc.)."""
    if not isinstance(text, str) or not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) >= min_ratio


def is_english(text: str) -> bool:
    """Robust English detector: fast ASCII reject followed by language detection."""
    if not is_ascii_candidate(text):
        return False
    # Strip URLs and mentions before detecting language
    text_for_detect = re.sub(r'https?://\S+', '', text)
    text_for_detect = re.sub(r'@\w+', '', text_for_detect).strip()
    if len(text_for_detect.split()) < 3:
        return False
    try:
        return detect(text_for_detect) == 'en'
    except Exception:
        return False


def extract_pairs(
    raw_csv_path: Path = DEFAULT_RAW_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    brand: str = DEFAULT_BRAND,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    chunksize: int = 250000,
) -> Path:
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{brand.lower()}_support_pairs.csv"

    print("=" * 60)
    print(f"Customer Support Pair Extraction for @{brand}")
    print(f"Source file : {raw_csv_path}")
    print(f"Target file : {output_file}")
    print(f"Max target  : {max_pairs} pairs")
    print("=" * 60)

    # -------------------------------------------------------
    # Step 1: Scan for brand responses (Pass 1)
    # -------------------------------------------------------
    print("\n[Pass 1/2] Scanning dataset for brand replies...")
    brand_chunks = []
    brand_usecols = ['tweet_id', 'author_id', 'inbound', 'text', 'in_response_to_tweet_id', 'created_at']

    for chunk in pd.read_csv(raw_csv_path, chunksize=chunksize, usecols=brand_usecols, low_memory=False):
        b_chunk = chunk[
            (chunk['author_id'] == brand) &
            (chunk['inbound'] == False) &
            (chunk['in_response_to_tweet_id'].notna())
        ]
        if not b_chunk.empty:
            brand_chunks.append(b_chunk)

    if not brand_chunks:
        raise ValueError(f"No replies found for brand: @{brand}")

    df_brand = pd.concat(brand_chunks, ignore_index=True)
    df_brand['in_response_to_tweet_id'] = df_brand['in_response_to_tweet_id'].astype(int)
    target_customer_ids = set(df_brand['in_response_to_tweet_id'])
    print(f"Found {len(df_brand):,} replies from @{brand} responding to {len(target_customer_ids):,} unique tweets.")

    # -------------------------------------------------------
    # Step 2: Scan for matching parent customer tweets (Pass 2)
    # -------------------------------------------------------
    print("\n[Pass 2/2] Scanning dataset for parent customer tweets...")
    cust_chunks = []
    cust_usecols = ['tweet_id', 'author_id', 'inbound', 'text', 'created_at']

    for chunk in pd.read_csv(raw_csv_path, chunksize=chunksize, usecols=cust_usecols, low_memory=False):
        c_chunk = chunk[chunk['tweet_id'].isin(target_customer_ids) & (chunk['inbound'] == True)]
        if not c_chunk.empty:
            cust_chunks.append(c_chunk)

    if not cust_chunks:
        raise ValueError("No matching parent customer tweets found.")

    df_customer = pd.concat(cust_chunks, ignore_index=True)
    print(f"Found {len(df_customer):,} matching customer inquiry tweets.")

    # -------------------------------------------------------
    # Step 3: Join customer inquiry -> Brand reply
    # -------------------------------------------------------
    print("\nMerging customer inquiries with brand responses...")
    merged = pd.merge(
        df_customer,
        df_brand,
        left_on='tweet_id',
        right_on='in_response_to_tweet_id',
        suffixes=('_customer', '_brand')
    )
    print(f"Total merged pairs: {len(merged):,}")

    # -------------------------------------------------------
    # Step 4: Text Cleaning & Deduplication
    # -------------------------------------------------------
    print("\nCleaning text and masking sensitive information...")
    merged['clean_text_customer'] = merged['text_customer'].apply(clean_tweet_text)
    merged['clean_text_brand'] = merged['text_brand'].apply(clean_tweet_text)

    # Filter by minimum length (discard trivial queries and short replies)
    merged = merged[
        (merged['clean_text_customer'].str.len() >= 25) &
        (merged['clean_text_brand'].str.len() >= 25)
    ]
    # Deduplicate on clean customer text to ensure rich variety
    merged = merged.drop_duplicates(subset=['clean_text_customer'])
    print(f"Pairs after length and uniqueness filtering: {len(merged):,}")

    # -------------------------------------------------------
    # Step 5: English Language Verification
    # -------------------------------------------------------
    print("\nFiltering for high-quality English conversations...")
    # Pre-filter candidate ASCII pairs first to avoid slow language detection on non-Latin text
    ascii_mask = (
        merged['clean_text_customer'].apply(is_ascii_candidate) &
        merged['clean_text_brand'].apply(is_ascii_candidate)
    )
    merged = merged[ascii_mask]

    # Detailed English language detection
    valid_pairs = []
    print(f"Verifying English language on candidate pairs (target: {max_pairs})...")
    for idx, row in merged.iterrows():
        c_text = row['clean_text_customer']
        b_text = row['clean_text_brand']
        if is_english(c_text) and is_english(b_text):
            valid_pairs.append(row)
            if len(valid_pairs) >= max_pairs:
                break

    df_final = pd.DataFrame(valid_pairs)
    if df_final.empty:
        raise ValueError("No valid English pairs found after filtering.")

    # Format output columns
    df_output = pd.DataFrame({
        'customer_tweet_id': df_final['tweet_id_customer'],
        'customer_text': df_final['clean_text_customer'],
        'brand_tweet_id': df_final['tweet_id_brand'],
        'brand_text': df_final['clean_text_brand'],
        'customer_created_at': df_final['created_at_customer'],
        'brand_created_at': df_final['created_at_brand'],
        'raw_customer_text': df_final['text_customer'],
        'raw_brand_text': df_final['text_brand'],
    })

    df_output.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved {len(df_output):,} clean English pairs to: {output_file}")

    # Display sample pairs
    print("\n" + "=" * 60)
    print("SAMPLE EXTRACTED PAIRS")
    print("=" * 60)
    for i in range(min(3, len(df_output))):
        print(f"\n[Sample {i + 1}]")
        print(f"Customer : {df_output.iloc[i]['customer_text']}")
        print(f"Brand    : {df_output.iloc[i]['brand_text']}")
    print("=" * 60)

    return output_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract customer-brand conversation pairs from twcs.csv")
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV, help="Path to raw twcs.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Path to output directory")
    parser.add_argument("--brand", type=str, default=DEFAULT_BRAND, help="Brand author_id (e.g. AmazonHelp)")
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS, help="Maximum pairs to extract")

    args = parser.parse_args()
    extract_pairs(
        raw_csv_path=args.raw_csv,
        output_dir=args.output_dir,
        brand=args.brand,
        max_pairs=args.max_pairs,
    )
