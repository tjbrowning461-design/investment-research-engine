import os
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


# =========================================================
# SETTINGS
# =========================================================

load_dotenv()

FILINGS_DIR = Path("filings")
FILINGS_DIR.mkdir(exist_ok=True)

SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "AI Financial Briefing Agent contact@example.com",
)

SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json,text/html",
}


# =========================================================
# SEC REQUEST HELPER
# =========================================================

def sec_get(
    url,
    timeout=30,
):

    response = requests.get(
        url,
        headers=SEC_HEADERS,
        timeout=timeout,
    )

    response.raise_for_status()

    return response


# =========================================================
# COMPANY CIK
# =========================================================

def get_company_cik(
    ticker,
):

    ticker = (
        ticker
        .strip()
        .upper()
    )

    url = (
        "https://www.sec.gov/files/"
        "company_tickers.json"
    )

    response = sec_get(
        url
    )

    companies = response.json()

    for company in companies.values():

        if (
            company.get(
                "ticker",
                "",
            ).upper()
            == ticker
        ):

            cik = str(
                company[
                    "cik_str"
                ]
            ).zfill(
                10
            )

            return cik

    raise ValueError(
        f"Could not find SEC CIK for ticker {ticker}."
    )


# =========================================================
# RECENT FILINGS
# =========================================================

def get_recent_filings(
    ticker,
):

    ticker = (
        ticker
        .strip()
        .upper()
    )

    cik = get_company_cik(
        ticker
    )

    url = (
        f"https://data.sec.gov/submissions/"
        f"CIK{cik}.json"
    )

    response = sec_get(
        url
    )

    data = response.json()

    recent = data[
        "filings"
    ][
        "recent"
    ]

    forms = recent[
        "form"
    ]

    accession_numbers = recent[
        "accessionNumber"
    ]

    filing_dates = recent[
        "filingDate"
    ]

    report_dates = recent[
        "reportDate"
    ]

    primary_documents = recent[
        "primaryDocument"
    ]

    result = {
        "10-Q": None,
        "10-K": None,
    }

    cik_without_leading_zeroes = str(
        int(
            cik
        )
    )

    for index, form in enumerate(
        forms
    ):

        if form not in [
            "10-Q",
            "10-K",
        ]:
            continue

        if result[
            form
        ] is not None:
            continue

        accession = accession_numbers[
            index
        ]

        accession_no_dashes = (
            accession.replace(
                "-",
                "",
            )
        )

        primary_document = (
            primary_documents[
                index
            ]
        )

        filing_url = (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{cik_without_leading_zeroes}/"
            f"{accession_no_dashes}/"
            f"{primary_document}"
        )

        result[
            form
        ] = {
            "ticker":
                ticker,

            "cik":
                cik,

            "form":
                form,

            "accession_number":
                accession,

            "filing_date":
                filing_dates[
                    index
                ],

            "report_date":
                report_dates[
                    index
                ],

            "primary_document":
                primary_document,

            "url":
                filing_url,
        }

        if (
            result[
                "10-Q"
            ]
            is not None
            and result[
                "10-K"
            ]
            is not None
        ):
            break

    return result


# =========================================================
# DOWNLOAD FILING HTML
# =========================================================

def download_filing_html(
    url,
):

    response = sec_get(
        url
    )

    return response.text


# =========================================================
# CLEAN FILING TEXT
# =========================================================

def clean_filing_text(
    html_text,
):

    soup = BeautifulSoup(
        html_text,
        "html.parser",
    )

    for tag in soup(
        [
            "script",
            "style",
            "noscript",
        ]
    ):

        tag.decompose()

    text = soup.get_text(
        separator=" "
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =========================================================
# SECTION EXTRACTION
# =========================================================

def extract_section(
    text,
    start_terms,
    end_terms,
    max_chars=30000,
):

    lower_text = text.lower()

    start_positions = []

    for term in start_terms:

        position = lower_text.find(
            term.lower()
        )

        if position != -1:

            start_positions.append(
                position
            )

    if not start_positions:

        return None

    start = min(
        start_positions
    )

    search_start = (
        start
        + 100
    )

    end_positions = []

    for term in end_terms:

        position = lower_text.find(
            term.lower(),
            search_start,
        )

        if position != -1:

            end_positions.append(
                position
            )

    if end_positions:

        end = min(
            end_positions
        )

    else:

        end = min(
            len(text),
            start
            + max_chars,
        )

    section = text[
        start:end
    ]

    if len(
        section
    ) > max_chars:

        section = section[
            :max_chars
        ]

    return section.strip()


# =========================================================
# 10-Q SECTION EXTRACTION
# =========================================================

def extract_10q_sections(
    text,
):

    management_discussion = extract_section(
        text,
        start_terms=[
            "Item 2. Management’s Discussion and Analysis",
            "Item 2. Management's Discussion and Analysis",
            "Management’s Discussion and Analysis of Financial Condition",
            "Management's Discussion and Analysis of Financial Condition",
        ],
        end_terms=[
            "Item 3. Quantitative and Qualitative Disclosures",
            "Item 4. Controls and Procedures",
        ],
        max_chars=30000,
    )

    risk_factors = extract_section(
        text,
        start_terms=[
            "Item 1A. Risk Factors",
            "Item 1A Risk Factors",
            "Risk Factors",
        ],
        end_terms=[
            "Item 2. Unregistered Sales",
            "Item 2 Unregistered Sales",
            "Item 3. Defaults",
            "Item 4. Mine Safety",
            "Item 5. Other Information",
            "Item 6. Exhibits",
        ],
        max_chars=30000,
    )

    liquidity = extract_section(
        text,
        start_terms=[
            "Liquidity and Capital Resources",
            "Liquidity and capital resources",
        ],
        end_terms=[
            "Critical Accounting Estimates",
            "Critical Accounting Policies",
            "Contractual Obligations",
            "Off-Balance Sheet Arrangements",
        ],
        max_chars=20000,
    )

    return {
        "management_discussion":
            management_discussion,

        "risk_factors":
            risk_factors,

        "liquidity":
            liquidity,
    }


# =========================================================
# LATEST 10-Q
# =========================================================

def get_latest_10q_text(
    ticker,
):

    ticker = (
        ticker
        .strip()
        .upper()
    )

    filings = get_recent_filings(
        ticker
    )

    filing = filings.get(
        "10-Q"
    )

    if not filing:

        return None

    html_text = download_filing_html(
        filing[
            "url"
        ]
    )

    full_text = clean_filing_text(
        html_text
    )

    sections = extract_10q_sections(
        full_text
    )

    output_file = (
        FILINGS_DIR
        / f"{ticker}_latest_10Q.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            full_text
        )

    return {
        "ticker":
            ticker,

        "filing_date":
            filing.get(
                "filing_date"
            ),

        "report_date":
            filing.get(
                "report_date"
            ),

        "url":
            filing.get(
                "url"
            ),

        "full_text":
            full_text,

        "sections":
            sections,

        "saved_to":
            str(
                output_file
            ),
    }


# =========================================================
# STANDALONE TEST
# =========================================================

def main():

    ticker = input(
        "Enter ticker: "
    ).strip().upper()

    print(
        f"\nLoading latest 10-Q for {ticker}..."
    )

    result = get_latest_10q_text(
        ticker
    )

    if not result:

        print(
            "No 10-Q found."
        )

        return

    print(
        "\nLatest 10-Q"
    )

    print(
        f"Report date: "
        f"{result['report_date']}"
    )

    print(
        f"Filed: "
        f"{result['filing_date']}"
    )

    print(
        f"URL: "
        f"{result['url']}"
    )

    print(
        f"Saved to: "
        f"{result['saved_to']}"
    )

    print(
        "\nSections found:"
    )

    for name, text in result[
        "sections"
    ].items():

        if text:

            print(
                f"- {name}: yes"
            )

        else:

            print(
                f"- {name}: no"
            )


if __name__ == "__main__":
    main()