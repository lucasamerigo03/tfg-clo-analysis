import os
import re
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# month name to number mapping used throughout date parsing functions
MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

PROCESSED_DIR = os.path.join("data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# Extraction functions - one per variable

def extract_manager_and_deal(filename):
    # parse manager and deal name from filename (format: MANAGER_DealName_YEAR-N.txt)
    base = filename.replace(".txt", "")
    parts = base.split("_", 1)
    manager = parts[0] if len(parts) > 0 else None
    deal_name = parts[1] if len(parts) > 1 else None
    return manager, deal_name


def extract_vintage(filename):
    # extract 4-digit year from filename
    match = re.search(r'(\d{4})-\d+', filename)
    if match:
        return int(match.group(1))
    return None


def extract_total_deal_size(text):
    # sum all tranche amounts from cover page; handles 3 formatting variants
    # excludes 'Up to' lines (undrawn facilities)
    TRANCHE_PATTERN = r'[€]\s*([\d,]+)\s*(?:Class\s+[A-Z]|Subordinated)'

    block_start = re.search(TRANCHE_PATTERN, text)

    if not block_start:
        block_start = re.search(
            r'^[A-Z](?:-\d)?\s+[€]\s*[\d,]+',
            text,
            re.MULTILINE
        )

    if not block_start:
        return None

    block = text[block_start.start(): block_start.start() + 3000]
    lines = block.split('\n')

    tranche_lines = []
    for line in lines:
        stripped = line.strip()
        if re.search(TRANCHE_PATTERN, stripped):
            tranche_lines.append(stripped)
        elif tranche_lines:
            break

    if len(tranche_lines) >= 4:
        total = 0
        for line in tranche_lines:
            if re.search(r'\bUp\s+to\b', line, re.IGNORECASE):
                continue
            m = re.search(TRANCHE_PATTERN, line)
            if m:
                value = float(m.group(1).replace(",", ""))
                # threshold filters out page numbers / reference numbers
                if value >= 100_000:
                    total += value
        if total > 0:
            return round(total / 1_000_000, 1)

    seen = {}
    for m in re.finditer(TRANCHE_PATTERN, block):
        raw = m.group(1).replace(",", "")
        seen[raw] = True

    if not seen:
        return None

    total = sum(float(r) for r in seen if float(r) >= 100_000)
    return round(total / 1_000_000, 1) if total > 0 else None


def parse_date_string(date_str):
    # converts '25 April 2030' or 'October 2021' to datetime
    date_str = date_str.strip()
    match = re.match(r'(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})', date_str)
    if match:
        day = int(match.group(1))
        month = MONTH_MAP.get(match.group(2).lower())
        year = int(match.group(3))
        if month:
            return datetime(year, month, day)
    match = re.match(r'([A-Za-z]+)\s+(\d{4})', date_str)
    if match:
        month = MONTH_MAP.get(match.group(1).lower())
        year = int(match.group(2))
        if month:
            return datetime(year, month, 1)
    return None


def extract_issue_date(text):
    # extract Issue Date from OC; used for precise period duration calculation
    patterns = [
        r'[Oo]n\s+or\s+about\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})\s*\(?(?:the\s+)?[""]?Issue\s+Date',
        r'[""]Issue\s+Date[""]\s+means\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
        r'Issue\s+Date[:\s]+(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
        r'dated\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})\s*\(?(?:the\s+)?[""]?Issue\s+Date',
        r'Issue\s+Date\s*\.{2,}\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            d = parse_date_string(m.group(1))
            if d:
                return d
    return None


def _calc_duration(end_date, issue_date, vintage, min_years=0.5, max_years=8.0):
    # duration in years from issue date (or vintage year as fallback)
    if issue_date:
        duration = round((end_date - issue_date).days / 365.25, 1)
    elif vintage:
        duration = round(end_date.year - vintage + (end_date.month - 1) / 12, 1)
    else:
        return None
    return duration if min_years <= duration <= max_years else None


def extract_reinvestment_period(text, vintage=None):
    # find reinvestment period end date from legal definition, return (date_str, years)
    block_match = re.search(
        r'["'"]Reinvestment\s+Period["'"]\s+means\s+the\s+period\s+from.{0,800}',
        text,
        re.DOTALL
    )

    if not block_match:
        block_match = re.search(
            r'Reinvestment\s+Period[.\s]+The\s+period\s+from.{0,800}',
            text,
            re.DOTALL
        )

    if not block_match:
        return None, None

    # limit to 800 chars - enough to capture the end date, avoids false positives
    block = block_match.group(0)[:800]

    date_patterns = [
        r'\(\s*(?:a|i)\s*\)\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
        r'[Bb]usiness\s+[Dd]ay\s+(?:prior\s+to|preceding|preceeding)[^;(]{0,80}?(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
        r'on\s+or\s+around\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
        r'[Pp]ayment\s+[Dd]ate\s+falling\s+on\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
        r'[Pp]ayment\s+[Dd]ate\s+falling\s+in\s+([A-Za-z]+\s+\d{4})',
    ]

    end_date = None
    for pattern in date_patterns:
        match = re.search(pattern, block)
        if match:
            end_date = parse_date_string(match.group(1))
            if end_date:
                break

    if end_date is None:
        return None, None

    issue_date = extract_issue_date(text)
    duration = _calc_duration(end_date, issue_date, vintage, min_years=1.0, max_years=8.0)

    return end_date.strftime("%Y-%m-%d"), duration


def extract_non_call_period(text, vintage=None):
    # extract non-call period duration in years from legal definition
    block_match = re.search(
        r'["'"]Non-Call\s+Period["'"]\s*'
        r'(?:means\s+(?:in\s+respect\s+of\s+the\s+Debt\s+)?'
        r'the\s+period\s+from.{0,600})',
        text,
        re.DOTALL
    )

    if not block_match:
        block_match = re.search(
            r'Non-Call\s+Period[^\n]{0,60}\n'
            r'(?:[^\n]{0,120}\n){0,2}'
            r'[^\n]*?(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
            text,
            re.DOTALL
        )
        if block_match:
            end_date = parse_date_string(block_match.group(1))
            if end_date:
                issue_date = extract_issue_date(text)
                duration = _calc_duration(end_date, issue_date, vintage,
                                          min_years=0.5, max_years=4.0)
                return duration
            return None

    if not block_match:
        return None

    block = re.sub(r'\n', ' ', block_match.group(0)[:600])

    date_patterns = [
        r'excluding[,\s]+(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
        r'[Pp]ayment\s+[Dd]ate\s+falling\s+in\s+([A-Za-z]+\s+\d{4})',
        r'falling\s+in\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
    ]

    for pattern in date_patterns:
        match = re.search(pattern, block)
        if match:
            end_date = parse_date_string(match.group(1))
            if end_date:
                issue_date = extract_issue_date(text)
                duration = _calc_duration(end_date, issue_date, vintage,
                                          min_years=0.5, max_years=4.0)
                if duration is not None:
                    return duration

    return None


def extract_ccc_limit(text):
    # extract CCC limit percentage from collateral quality tests table
    VALUE = r'(\d+(?:\.\d+)?)\s*(?:%|per\s+cent\.?)' 

    patterns = [
        rf'CCC\s+Obligations?\s+N/A\s+{VALUE}',
        rf'S&P\s+CCC\s+Obligations?\s+N/A\s+{VALUE}',
        rf'CCC\+[""]?\s+or\s+below[^.]*?(?:may\s+not\s+exceed|not\s+more\s+than)\s+{VALUE}',
        rf'maximum\s+(?:of\s+)?{VALUE}[^.]*?CCC',
        rf'CCC[^.]*?{VALUE}',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            # sanity check: any CCC limit outside 1-30% range is a parsing error
            if 1 <= value <= 30:
                return value

    return None


def extract_oc_ratio_class_a(text):
    # extract required OC (par value) ratio for Class A from coverage tests table
    VALUE_PCT = r'(\d{2,3}(?:\.\d+)?)\s*(?:%|per\s+cent\.?)' 
    VALUE_BARE = r'(\d{2,3}(?:\.\d+)?)' 
    CLASS = r'(?:A(?:/B)?|Senior)'

    patterns = [
        rf'Required\s+(?:Par\s+Value|Principal\s+Coverage)\s+Ratio\s*\n\s*{CLASS}\s+{VALUE_PCT}',
        rf'Class\s+Required\s+(?:Par\s+Value|Principal\s+Coverage)\s*\n\s*{CLASS}\s+{VALUE_PCT}',
        rf'Class\s+Required\s+Par\s+Value\s*\n\s*{CLASS}\s+{VALUE_PCT}',
        rf'{CLASS}\s+{VALUE_PCT}\s*\n\s*[BC]\s+\d',
        rf'{CLASS}\s*\.{{2,}}\s*{VALUE_PCT}',
        rf'Required\s+Par\s+Value\s+Ratio\s*\(%\)\s*\n\s*{CLASS}\s+{VALUE_BARE}',
        rf'Required\s+Par\s*\n\s*Class\s+Value\s+Ratio[^\n]*\n\s*{CLASS}\s+{VALUE_BARE}',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            # realistic OC ratio range for CLO Class A tranches
            if 110 <= value <= 145:
                return value

    return None


def extract_num_tranches(text):
    # count rated tranches from cover page (excludes sub notes)
    pattern_clean = re.findall(
        r'[€]\s*[\d,]+\s+Class\s+([A-F](?:-\d)?)\s+\w',
        text
    )

    if pattern_clean:
        return len(set(pattern_clean))

    pattern_table = re.findall(
        r'^([A-F](?:-\d)?)\s+[€]\s*[\d,]+',
        text,
        re.MULTILINE
    )

    if pattern_table:
        return len(set(pattern_table))

    return None


def extract_class_a_size(text):
    # extract Class A tranche size in EUR mn
    anchors = list(re.finditer(
        r'(?:designated activity company|besloten vennootschap|'
        r'private company with limited liability|société anonyme|'
        r'limited liability company)[^\n]{0,300}',
        text, re.IGNORECASE | re.DOTALL
    ))

    raw_block = None
    for m in anchors:
        candidate = text[m.start(): m.start() + 3000]
        if re.search(r'[€]\s*[\d,]{6,}', candidate):
            raw_block = candidate
            break

    if raw_block is None:
        fb = re.search(r'[€]\s*[\d,]+\s*Class\s+[A-Z]', text)
        if not fb:
            return None
        raw_block = text[fb.start(): fb.start() + 3000]

    sub_match = re.search(r'Subordinated\s+Notes', raw_block)
    block = raw_block[:sub_match.end()] if sub_match else raw_block[:2000]

    lines = block.split('\n')
    total_line = 0.0
    found_line = False
    for line in lines:
        if re.search(r'\bUp\s+to\b', line, re.IGNORECASE):
            continue
        m = re.search(
            r'[€]\s*([\d,]+)\s*Class\s+A(?:[-\s]\d[A-Z]?|[-]\d)?\s+\w*\s*Secured',
            line
        )
        if m:
            total_line += float(m.group(1).replace(",", ""))
            found_line = True

    if found_line and total_line > 0:
        return round(total_line / 1_000_000, 1) if total_line > 1000 else total_line

    match_split = re.search(
        r'^(A(?:-1[A-Z]?)?)\s+[€]\s*([\d,]+)[^\n]*\n\s*(\d{2,3})\b',
        block,
        re.MULTILINE
    )
    if match_split:
        raw = match_split.group(2).replace(",", "") + match_split.group(3)
        value = float(raw)
        return round(value / 1_000_000, 1) if value > 1000 else value

    matches_table = re.findall(
        r'^(A(?:-1[A-Z]?)?)\s+[€]\s*([\d,]+)',
        block,
        re.MULTILINE
    )
    if matches_table:
        total = sum(float(v.replace(",", "")) for _, v in matches_table)
        return round(total / 1_000_000, 1) if total > 1000 else total

    return None


def extract_sub_notes_size(text):
    # extract subordinated notes size in EUR mn
    anchors = list(re.finditer(
        r'(?:designated activity company|besloten vennootschap|'
        r'private company with limited liability|société anonyme|'
        r'limited liability company)[^\n]{0,300}',
        text, re.IGNORECASE | re.DOTALL
    ))

    cover_block = None
    for m in anchors:
        candidate = text[m.start(): m.start() + 3000]
        if re.search(r'[€]\s*[\d,]{6,}', candidate):
            cover_block = candidate
            break

    if cover_block is None:
        cover_block = text[:3000]

    first_mx = re.search(
        r'[€]\s*[\d,]+\s+Class\s+M-\d+\s+Subordinated\s+Notes\s+due\s+\d{4}',
        cover_block
    )
    if first_mx:
        mx_block = cover_block[first_mx.start():]
        lines = mx_block.split('\n')
        mx_values = []
        for line in lines:
            m = re.match(
                r'[€]\s*([\d,]+)\s+Class\s+M-\d+\s+Subordinated\s+Notes\s+due\s+\d{4}\s*$',
                line.strip()
            )
            if m:
                mx_values.append(float(m.group(1).replace(",", "")))
            elif mx_values:
                break
        if mx_values:
            total = sum(mx_values)
            return round(total / 1_000_000, 2) if total > 1000 else total

    match = re.search(r'[€]\s*([\d,]+)\s+Subordinated\s+Notes', text)
    if match:
        value = float(match.group(1).replace(",", ""))
        return round(value / 1_000_000, 1) if value > 1000 else value

    match = re.search(r'Subordinated\s+Notes[^\n€]{0,40}[€]\s*([\d,]+)', text)
    if match:
        value = float(match.group(1).replace(",", ""))
        return round(value / 1_000_000, 1) if value > 1000 else value

    match = re.search(
        r'Subordinated\s+(?:Notes\s+)?[€]\s*([\d,]+)\n\s*(\d+)',
        text, re.MULTILINE
    )
    if match:
        raw = match.group(1).replace(",", "") + match.group(2)
        value = float(raw)
        return round(value / 1_000_000, 1) if value > 1000 else value

    return None




def parse_single_doc(filename, text):
    # run all extraction functions on a single document
    manager, deal_name = extract_manager_and_deal(filename)
    vintage = extract_vintage(filename)
    reinvestment_end, reinvestment_duration = extract_reinvestment_period(text, vintage=vintage)

    return {
        "filename":              filename,
        "manager":               manager,
        "deal_name":             deal_name,
        "vintage":               vintage,
        "total_deal_size_mn":    extract_total_deal_size(text),
        "reinvestment_end_date": reinvestment_end,
        "reinvestment_period":   reinvestment_duration,
        "non_call_period":       extract_non_call_period(text, vintage=vintage),
        "ccc_limit_pct":         extract_ccc_limit(text),
        "oc_ratio_class_a":      extract_oc_ratio_class_a(text),
        "num_tranches":          extract_num_tranches(text),
        "class_a_size_mn":       extract_class_a_size(text),
        "sub_notes_size_mn":     extract_sub_notes_size(text),
    }


def main():
    txt_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".txt")]
    print(f"Documents found: {len(txt_files)}\n")

    records = []
    for filename in tqdm(txt_files, desc="Parsing OCs"):
        filepath = os.path.join(PROCESSED_DIR, filename)
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
        record = parse_single_doc(filename, text)
        records.append(record)

    df = pd.DataFrame(records)

    output_path = os.path.join("data", "processed", "clo_dataset_raw.csv")
    df.to_csv(output_path, index=False)

    # coverage report - useful for spotting which patterns need improvement
    print("\n--- Extraction coverage ---")
    for col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  {col:<25} {non_null:>2}/{len(df)} extracted")

    print(f"\nDataset saved to: {output_path}")


if __name__ == "__main__":
    main()
