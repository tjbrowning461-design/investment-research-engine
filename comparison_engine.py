import os
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv


# =========================================================
# SETTINGS
# =========================================================

load_dotenv()

SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "Company Comparison Engine contact@example.com",
)

SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}

MAX_COMPANIES = 10


# =========================================================
# BASIC HELPERS
# =========================================================

def safe_divide(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def growth(current, previous):
    if current is None or previous in (None, 0):
        return None
    return (current - previous) / abs(previous)


def parse_date(value):
    if not value:
        return None

    try:
        return datetime.strptime(
            value,
            "%Y-%m-%d",
        )

    except Exception:
        return None


def to_float(value):
    if value is None:
        return None

    try:
        return float(value)

    except Exception:
        return None


def duration_days(entry):
    start = parse_date(
        entry.get(
            "start"
        )
    )

    end = parse_date(
        entry.get(
            "end"
        )
    )

    if start is None or end is None:
        return None

    return (
        end - start
    ).days


def near_same_date(
    date_a,
    date_b,
    tolerance=15,
):
    a = parse_date(
        date_a
    )

    b = parse_date(
        date_b
    )

    if a is None or b is None:
        return False

    return abs(
        (
            a - b
        ).days
    ) <= tolerance


# =========================================================
# SEC REQUESTS / CACHES
# =========================================================

_CIK_CACHE = None
_COMPANY_FACTS_CACHE = {}
_SUBMISSIONS_CACHE = {}


def sec_get_json(url):
    response = requests.get(
        url,
        headers=SEC_HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def get_company_cik(ticker):
    global _CIK_CACHE

    ticker = (
        ticker
        .strip()
        .upper()
    )

    if _CIK_CACHE is None:
        _CIK_CACHE = sec_get_json(
            "https://www.sec.gov/files/company_tickers.json"
        )

    for company in _CIK_CACHE.values():

        if (
            company.get(
                "ticker",
                "",
            ).upper()
            == ticker
        ):

            return str(
                company[
                    "cik_str"
                ]
            ).zfill(
                10
            )

    raise ValueError(
        f"SEC CIK not found for {ticker}."
    )


def get_company_facts(ticker):
    ticker = (
        ticker
        .strip()
        .upper()
    )

    if ticker not in _COMPANY_FACTS_CACHE:

        cik = get_company_cik(
            ticker
        )

        url = (
            "https://data.sec.gov/api/xbrl/companyfacts/"
            f"CIK{cik}.json"
        )

        _COMPANY_FACTS_CACHE[
            ticker
        ] = sec_get_json(
            url
        )

    return _COMPANY_FACTS_CACHE[
        ticker
    ]


def get_submissions(ticker):
    ticker = (
        ticker
        .strip()
        .upper()
    )

    if ticker not in _SUBMISSIONS_CACHE:

        cik = get_company_cik(
            ticker
        )

        url = (
            f"https://data.sec.gov/submissions/"
            f"CIK{cik}.json"
        )

        _SUBMISSIONS_CACHE[
            ticker
        ] = sec_get_json(
            url
        )

    return _SUBMISSIONS_CACHE[
        ticker
    ]


# =========================================================
# LATEST SEC REPORTING PERIOD
# =========================================================

def get_latest_reporting_period(
    ticker
):
    submissions = get_submissions(
        ticker
    )

    recent = (
        submissions
        .get(
            "filings",
            {},
        )
        .get(
            "recent",
            {},
        )
    )

    forms = recent.get(
        "form",
        [],
    )

    report_dates = recent.get(
        "reportDate",
        [],
    )

    filing_dates = recent.get(
        "filingDate",
        [],
    )

    accession_numbers = recent.get(
        "accessionNumber",
        [],
    )

    primary_documents = recent.get(
        "primaryDocument",
        [],
    )

    candidates = []

    for index, form in enumerate(
        forms
    ):

        if form not in (
            "10-Q",
            "10-K",
        ):
            continue

        report_date = (
            report_dates[
                index
            ]
            if index < len(
                report_dates
            )
            else None
        )

        filing_date = (
            filing_dates[
                index
            ]
            if index < len(
                filing_dates
            )
            else None
        )

        if not report_date:
            continue

        candidates.append(
            {
                "form":
                    form,

                "report_date":
                    report_date,

                "filing_date":
                    filing_date,

                "accession_number":
                    (
                        accession_numbers[
                            index
                        ]
                        if index
                        < len(
                            accession_numbers
                        )
                        else None
                    ),

                "primary_document":
                    (
                        primary_documents[
                            index
                        ]
                        if index
                        < len(
                            primary_documents
                        )
                        else None
                    ),
            }
        )

    if not candidates:

        raise ValueError(
            "No recent 10-Q or 10-K found."
        )

    candidates.sort(
        key=lambda item: (
            item.get(
                "report_date"
            )
            or "",

            item.get(
                "filing_date"
            )
            or "",
        ),
        reverse=True,
    )

    return candidates[
        0
    ]


# =========================================================
# COMPANY FACTS TAG HELPERS
# =========================================================

def all_tag_entries(
    company_facts,
    tags,
    preferred_units=None,
):
    us_gaap = (
        company_facts
        .get(
            "facts",
            {},
        )
        .get(
            "us-gaap",
            {},
        )
    )

    collected = []

    for tag_priority, tag in enumerate(
        tags
    ):

        fact = us_gaap.get(
            tag
        )

        if not fact:
            continue

        units = fact.get(
            "units",
            {},
        )

        unit_names = []

        if preferred_units:

            for unit in preferred_units:

                if unit in units:
                    unit_names.append(
                        unit
                    )

        for unit in units:

            if unit not in unit_names:
                unit_names.append(
                    unit
                )

        for unit in unit_names:

            for entry in units.get(
                unit,
                [],
            ):

                collected.append(
                    {
                        "entry":
                            entry,

                        "tag":
                            tag,

                        "unit":
                            unit,

                        "tag_priority":
                            tag_priority,
                    }
                )

    return collected


def choose_best_entry(
    candidates
):
    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[
                "entry"
            ].get(
                "filed",
                "",
            ),

            -item.get(
                "tag_priority",
                999,
            ),
        ),
        reverse=True,
    )

    return candidates[
        0
    ]


def find_quarter_fact(
    company_facts,
    tags,
    target_end,
    preferred_units=None,
    allowed_forms=(
        "10-Q",
        "10-K",
    ),
):
    candidates = []

    for item in all_tag_entries(
        company_facts,
        tags,
        preferred_units,
    ):

        entry = item[
            "entry"
        ]

        if entry.get(
            "form"
        ) not in allowed_forms:
            continue

        if not near_same_date(
            entry.get(
                "end"
            ),
            target_end,
            tolerance=7,
        ):
            continue

        days = duration_days(
            entry
        )

        if (
            days is None
            or not (
                65
                <= days
                <= 120
            )
        ):
            continue

        if to_float(
            entry.get(
                "val"
            )
        ) is None:
            continue

        candidates.append(
            item
        )

    best = choose_best_entry(
        candidates
    )

    if not best:
        return None

    return {
        "value":
            to_float(
                best[
                    "entry"
                ].get(
                    "val"
                )
            ),

        "entry":
            best[
                "entry"
            ],

        "tag":
            best[
                "tag"
            ],

        "unit":
            best[
                "unit"
            ],
    }


def find_annual_fact(
    company_facts,
    tags,
    target_end,
    preferred_units=None,
):
    candidates = []

    for item in all_tag_entries(
        company_facts,
        tags,
        preferred_units,
    ):

        entry = item[
            "entry"
        ]

        if entry.get(
            "form"
        ) != "10-K":
            continue

        if not near_same_date(
            entry.get(
                "end"
            ),
            target_end,
            tolerance=7,
        ):
            continue

        days = duration_days(
            entry
        )

        if (
            days is None
            or not (
                300
                <= days
                <= 390
            )
        ):
            continue

        if to_float(
            entry.get(
                "val"
            )
        ) is None:
            continue

        candidates.append(
            item
        )

    best = choose_best_entry(
        candidates
    )

    if not best:
        return None

    return {
        "value":
            to_float(
                best[
                    "entry"
                ].get(
                    "val"
                )
            ),

        "entry":
            best[
                "entry"
            ],

        "tag":
            best[
                "tag"
            ],

        "unit":
            best[
                "unit"
            ],
    }


def find_ytd_fact(
    company_facts,
    tags,
    target_end,
    preferred_units=None,
    min_days=130,
    max_days=300,
):
    candidates = []

    for item in all_tag_entries(
        company_facts,
        tags,
        preferred_units,
    ):

        entry = item[
            "entry"
        ]

        if entry.get(
            "form"
        ) != "10-Q":
            continue

        if not near_same_date(
            entry.get(
                "end"
            ),
            target_end,
            tolerance=7,
        ):
            continue

        days = duration_days(
            entry
        )

        if (
            days is None
            or not (
                min_days
                <= days
                <= max_days
            )
        ):
            continue

        if to_float(
            entry.get(
                "val"
            )
        ) is None:
            continue

        candidates.append(
            item
        )

    best = choose_best_entry(
        candidates
    )

    if not best:
        return None

    return {
        "value":
            to_float(
                best[
                    "entry"
                ].get(
                    "val"
                )
            ),

        "entry":
            best[
                "entry"
            ],

        "tag":
            best[
                "tag"
            ],

        "unit":
            best[
                "unit"
            ],
    }


def find_point_fact(
    company_facts,
    tags,
    target_end,
    preferred_units=None,
):
    candidates = []

    for item in all_tag_entries(
        company_facts,
        tags,
        preferred_units,
    ):

        entry = item[
            "entry"
        ]

        if entry.get(
            "form"
        ) not in (
            "10-Q",
            "10-K",
        ):
            continue

        if entry.get(
            "start"
        ):
            continue

        if not near_same_date(
            entry.get(
                "end"
            ),
            target_end,
            tolerance=7,
        ):
            continue

        if to_float(
            entry.get(
                "val"
            )
        ) is None:
            continue

        candidates.append(
            item
        )

    best = choose_best_entry(
        candidates
    )

    if not best:
        return None

    return {
        "value":
            to_float(
                best[
                    "entry"
                ].get(
                    "val"
                )
            ),

        "entry":
            best[
                "entry"
            ],

        "tag":
            best[
                "tag"
            ],

        "unit":
            best[
                "unit"
            ],
    }


# =========================================================
# Q4 DERIVATION
# =========================================================

def prior_single_quarters_before_end(
    company_facts,
    tags,
    target_end,
    preferred_units=None,
    count=3,
):
    target = parse_date(
        target_end
    )

    if target is None:
        return []

    by_end = {}

    for item in all_tag_entries(
        company_facts,
        tags,
        preferred_units,
    ):

        entry = item[
            "entry"
        ]

        if entry.get(
            "form"
        ) != "10-Q":
            continue

        end = parse_date(
            entry.get(
                "end"
            )
        )

        days = duration_days(
            entry
        )

        value = to_float(
            entry.get(
                "val"
            )
        )

        if (
            end is None
            or days is None
            or value is None
        ):
            continue

        if end >= target:
            continue

        if not (
            65
            <= days
            <= 120
        ):
            continue

        gap = (
            target - end
        ).days

        if gap > 330:
            continue

        end_key = entry.get(
            "end"
        )

        current = by_end.get(
            end_key
        )

        if (
            current is None
            or entry.get(
                "filed",
                "",
            )
            > current[
                "entry"
            ].get(
                "filed",
                "",
            )
            or (
                entry.get(
                    "filed",
                    "",
                )
                == current[
                    "entry"
                ].get(
                    "filed",
                    "",
                )
                and item[
                    "tag_priority"
                ]
                < current[
                    "tag_priority"
                ]
            )
        ):

            by_end[
                end_key
            ] = item

    ordered = sorted(
        by_end.values(),
        key=lambda item: item[
            "entry"
        ].get(
            "end",
            "",
        ),
        reverse=True,
    )

    return ordered[
        :count
    ]


def derive_q4_flow(
    company_facts,
    tags,
    target_end,
    preferred_units=None,
):
    annual = find_annual_fact(
        company_facts,
        tags,
        target_end,
        preferred_units,
    )

    if annual is None:
        return None

    quarters = prior_single_quarters_before_end(
        company_facts,
        tags,
        target_end,
        preferred_units,
        count=3,
    )

    if len(
        quarters
    ) < 3:
        return None

    quarter_values = [
        to_float(
            item[
                "entry"
            ].get(
                "val"
            )
        )
        for item in quarters
    ]

    if any(
        value is None
        for value in quarter_values
    ):
        return None

    return {
        "value":
            annual[
                "value"
            ]
            - sum(
                quarter_values
            ),

        "source":
            (
                "SEC 10-K derived Q4 "
                "(annual less Q1-Q3)"
            ),

        "tag":
            annual[
                "tag"
            ],
    }


def derive_q4_cash_flow(
    company_facts,
    tags,
    target_end,
    preferred_units=None,
):
    annual = find_annual_fact(
        company_facts,
        tags,
        target_end,
        preferred_units,
    )

    if annual is None:
        return None

    target = parse_date(
        target_end
    )

    if target is None:
        return None

    candidates = []

    for item in all_tag_entries(
        company_facts,
        tags,
        preferred_units,
    ):

        entry = item[
            "entry"
        ]

        if entry.get(
            "form"
        ) != "10-Q":
            continue

        end = parse_date(
            entry.get(
                "end"
            )
        )

        days = duration_days(
            entry
        )

        value = to_float(
            entry.get(
                "val"
            )
        )

        if (
            end is None
            or days is None
            or value is None
        ):
            continue

        if end >= target:
            continue

        if not (
            220
            <= days
            <= 310
        ):
            continue

        gap = (
            target - end
        ).days

        if gap > 140:
            continue

        candidates.append(
            item
        )

    best = choose_best_entry(
        candidates
    )

    if best is None:
        return None

    nine_month_value = to_float(
        best[
            "entry"
        ].get(
            "val"
        )
    )

    if nine_month_value is None:
        return None

    return {
        "value":
            annual[
                "value"
            ]
            - nine_month_value,

        "source":
            (
                "SEC 10-K derived Q4 "
                "(annual less 9M YTD)"
            ),

        "tag":
            annual[
                "tag"
            ],
    }


# =========================================================
# FLOW RESOLUTION
# =========================================================

def resolve_flow_metric(
    company_facts,
    tags,
    target_end,
    form,
    preferred_units=None,
    allow_q4_derivation=True,
):
    direct = find_quarter_fact(
        company_facts,
        tags,
        target_end,
        preferred_units,
    )

    if direct is not None:

        return {
            "value":
                direct[
                    "value"
                ],

            "source":
                "SEC Company Facts direct quarter",

            "tag":
                direct[
                    "tag"
                ],
        }

    if (
        form == "10-K"
        and allow_q4_derivation
    ):

        derived = derive_q4_flow(
            company_facts,
            tags,
            target_end,
            preferred_units,
        )

        if derived is not None:
            return derived

    return None


# =========================================================
# PRIOR-YEAR MATCHING
# =========================================================

def find_prior_year_reporting_period(
    ticker,
    latest_report_date,
    latest_form,
):
    submissions = get_submissions(
        ticker
    )

    recent = (
        submissions
        .get(
            "filings",
            {},
        )
        .get(
            "recent",
            {},
        )
    )

    forms = recent.get(
        "form",
        [],
    )

    report_dates = recent.get(
        "reportDate",
        [],
    )

    filing_dates = recent.get(
        "filingDate",
        [],
    )

    latest_dt = parse_date(
        latest_report_date
    )

    if latest_dt is None:
        return None

    candidates = []

    for index, form in enumerate(
        forms
    ):

        if form != latest_form:
            continue

        report_date = (
            report_dates[
                index
            ]
            if index
            < len(
                report_dates
            )
            else None
        )

        report_dt = parse_date(
            report_date
        )

        if report_dt is None:
            continue

        difference = (
            latest_dt
            - report_dt
        ).days

        if (
            330
            <= difference
            <= 400
        ):

            candidates.append(
                {
                    "difference":
                        abs(
                            difference
                            - 365
                        ),

                    "form":
                        form,

                    "report_date":
                        report_date,

                    "filing_date":
                        (
                            filing_dates[
                                index
                            ]
                            if index
                            < len(
                                filing_dates
                            )
                            else None
                        ),
                }
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[
                "difference"
            ],

            item.get(
                "filing_date"
            )
            or "",
        )
    )

    return candidates[
        0
    ]


# =========================================================
# CASH FLOW RESOLUTION
# =========================================================

def resolve_quarterly_cash_flow(
    company_facts,
    tags,
    target_end,
    form,
    preferred_units=None,
):
    direct = find_quarter_fact(
        company_facts,
        tags,
        target_end,
        preferred_units,
    )

    if direct is not None:

        return {
            "value":
                direct[
                    "value"
                ],

            "source":
                "SEC Company Facts direct quarter",

            "tag":
                direct[
                    "tag"
                ],
        }

    if form == "10-K":

        return derive_q4_cash_flow(
            company_facts,
            tags,
            target_end,
            preferred_units,
        )

    ytd = find_ytd_fact(
        company_facts,
        tags,
        target_end,
        preferred_units,
        min_days=130,
        max_days=300,
    )

    if ytd is None:
        return None

    target_dt = parse_date(
        target_end
    )

    if target_dt is None:
        return None

    ytd_days = duration_days(
        ytd[
            "entry"
        ]
    )

    candidates = []

    for item in all_tag_entries(
        company_facts,
        tags,
        preferred_units,
    ):

        entry = item[
            "entry"
        ]

        if entry.get(
            "form"
        ) != "10-Q":
            continue

        end = parse_date(
            entry.get(
                "end"
            )
        )

        days = duration_days(
            entry
        )

        value = to_float(
            entry.get(
                "val"
            )
        )

        if (
            end is None
            or days is None
            or value is None
        ):
            continue

        if end >= target_dt:
            continue

        if days >= ytd_days:
            continue

        gap = (
            target_dt
            - end
        ).days

        if gap > 140:
            continue

        candidates.append(
            item
        )

    previous = choose_best_entry(
        candidates
    )

    if previous is None:
        return None

    previous_value = to_float(
        previous[
            "entry"
        ].get(
            "val"
        )
    )

    if previous_value is None:
        return None

    return {
        "value":
            ytd[
                "value"
            ]
            - previous_value,

        "source":
            (
                "SEC Company Facts "
                "derived quarter from YTD"
            ),

        "tag":
            ytd[
                "tag"
            ],
    }


# =========================================================
# YAHOO EXACT-PERIOD FALLBACK
# =========================================================

def yahoo_value_for_period(
    statement,
    names,
    target_end,
    tolerance=15,
):
    if (
        statement is None
        or statement.empty
    ):
        return None

    target_date = pd.Timestamp(
        target_end
    )

    closest_column = None
    closest_difference = None

    for column in statement.columns:

        column_date = pd.Timestamp(
            column
        )

        difference = abs(
            (
                target_date
                - column_date
            ).days
        )

        if (
            closest_difference is None
            or difference
            < closest_difference
        ):

            closest_difference = (
                difference
            )

            closest_column = column

    if (
        closest_column is None
        or closest_difference is None
        or closest_difference
        > tolerance
    ):
        return None

    for name in names:

        if name not in statement.index:
            continue

        value = statement.loc[
            name,
            closest_column,
        ]

        if pd.isna(
            value
        ):
            continue

        result = to_float(
            value
        )

        if result is not None:
            return result

    return None


def yahoo_income_fallback(
    ticker_obj,
    target_end,
    names,
):
    try:

        statement = (
            ticker_obj
            .quarterly_income_stmt
        )

    except Exception:

        return None

    return yahoo_value_for_period(
        statement,
        names,
        target_end,
    )


def yahoo_cashflow_fallback(
    ticker_obj,
    target_end,
    names,
):
    try:

        statement = (
            ticker_obj
            .quarterly_cashflow
        )

    except Exception:

        return None

    return yahoo_value_for_period(
        statement,
        names,
        target_end,
    )


def yahoo_balance_fallback(
    ticker_obj,
    target_end,
    names,
):
    try:

        statement = (
            ticker_obj
            .quarterly_balance_sheet
        )

    except Exception:

        return None

    return yahoo_value_for_period(
        statement,
        names,
        target_end,
    )


# =========================================================
# TAGS
# =========================================================

REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "Revenues",
    "SalesRevenueGoodsNet",
]

GROSS_PROFIT_TAGS = [
    "GrossProfit",
]

OPERATING_INCOME_TAGS = [
    "OperatingIncomeLoss",
]

NET_INCOME_TAGS = [
    "NetIncomeLoss",
    "ProfitLoss",
]

PRETAX_INCOME_TAGS = [
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
]

EPS_TAGS = [
    "EarningsPerShareDiluted",
]

OCF_TAGS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]

CAPEX_TAGS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsForAdditionsToPropertyPlantAndEquipment",
]

CASH_TAGS = [
    "CashAndCashEquivalentsAtCarryingValue",
]

CURRENT_ASSETS_TAGS = [
    "AssetsCurrent",
]

CURRENT_LIABILITIES_TAGS = [
    "LiabilitiesCurrent",
]

EQUITY_TAGS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]


# =========================================================
# DEBT
# =========================================================

def resolve_total_debt(
    company_facts,
    ticker_obj,
    target_end,
):
    yahoo_debt = yahoo_balance_fallback(
        ticker_obj,
        target_end,
        [
            "Total Debt",
        ],
    )

    if yahoo_debt is not None:

        return (
            yahoo_debt,
            "Yahoo Finance exact-period",
        )

    return (
        None,
        "Unavailable",
    )


# =========================================================
# RESOLVE ALL METRICS FOR ONE PERIOD
# =========================================================

def resolve_period_metrics(
    company_facts,
    ticker_obj,
    period,
):
    target_end = period[
        "report_date"
    ]

    form = period[
        "form"
    ]

    revenue_result = resolve_flow_metric(
        company_facts,
        REVENUE_TAGS,
        target_end,
        form,
        preferred_units=[
            "USD"
        ],
    )

    if revenue_result is None:

        value = yahoo_income_fallback(
            ticker_obj,
            target_end,
            [
                "Total Revenue",
                "Operating Revenue",
            ],
        )

        if value is not None:

            revenue_result = {
                "value":
                    value,

                "source":
                    "Yahoo Finance exact-period fallback",

                "tag":
                    None,
            }

    gross_profit_result = resolve_flow_metric(
        company_facts,
        GROSS_PROFIT_TAGS,
        target_end,
        form,
        preferred_units=[
            "USD"
        ],
    )

    if gross_profit_result is None:

        value = yahoo_income_fallback(
            ticker_obj,
            target_end,
            [
                "Gross Profit",
            ],
        )

        if value is not None:

            gross_profit_result = {
                "value":
                    value,

                "source":
                    "Yahoo Finance exact-period fallback",

                "tag":
                    None,
            }

    operating_income_result = resolve_flow_metric(
        company_facts,
        OPERATING_INCOME_TAGS,
        target_end,
        form,
        preferred_units=[
            "USD"
        ],
    )

    if operating_income_result is None:

        value = yahoo_income_fallback(
            ticker_obj,
            target_end,
            [
                "Operating Income",
            ],
        )

        if value is not None:

            operating_income_result = {
                "value":
                    value,

                "source":
                    "Yahoo Finance exact-period fallback",

                "tag":
                    None,
            }

    net_income_result = resolve_flow_metric(
        company_facts,
        NET_INCOME_TAGS,
        target_end,
        form,
        preferred_units=[
            "USD"
        ],
    )

    if net_income_result is None:

        value = yahoo_income_fallback(
            ticker_obj,
            target_end,
            [
                "Net Income",
                "Net Income Common Stockholders",
            ],
        )

        if value is not None:

            net_income_result = {
                "value":
                    value,

                "source":
                    "Yahoo Finance exact-period fallback",

                "tag":
                    None,
            }

    pretax_result = resolve_flow_metric(
        company_facts,
        PRETAX_INCOME_TAGS,
        target_end,
        form,
        preferred_units=[
            "USD"
        ],
    )

    if pretax_result is None:

        value = yahoo_income_fallback(
            ticker_obj,
            target_end,
            [
                "Pretax Income",
                "Pre Tax Income",
            ],
        )

        if value is not None:

            pretax_result = {
                "value":
                    value,

                "source":
                    "Yahoo Finance exact-period fallback",

                "tag":
                    None,
            }

    eps_direct = find_quarter_fact(
        company_facts,
        EPS_TAGS,
        target_end,
        preferred_units=[
            "USD/shares"
        ],
    )

    diluted_eps = (
        eps_direct[
            "value"
        ]
        if eps_direct
        else None
    )

    eps_source = (
        "SEC Company Facts direct quarter"
        if eps_direct
        else None
    )

    if diluted_eps is None:

        diluted_eps = yahoo_income_fallback(
            ticker_obj,
            target_end,
            [
                "Diluted EPS",
                "Diluted EPS Continuous Operations",
            ],
        )

        if diluted_eps is not None:

            eps_source = (
                "Yahoo Finance exact-period fallback"
            )

    ocf_result = resolve_quarterly_cash_flow(
        company_facts,
        OCF_TAGS,
        target_end,
        form,
        preferred_units=[
            "USD"
        ],
    )

    if ocf_result is None:

        value = yahoo_cashflow_fallback(
            ticker_obj,
            target_end,
            [
                "Operating Cash Flow",
                "Total Cash From Operating Activities",
            ],
        )

        if value is not None:

            ocf_result = {
                "value":
                    value,

                "source":
                    "Yahoo Finance exact-period fallback",

                "tag":
                    None,
            }

    capex_result = resolve_quarterly_cash_flow(
        company_facts,
        CAPEX_TAGS,
        target_end,
        form,
        preferred_units=[
            "USD"
        ],
    )

    if capex_result is None:

        value = yahoo_cashflow_fallback(
            ticker_obj,
            target_end,
            [
                "Capital Expenditure",
                "Capital Expenditures",
            ],
        )

        if value is not None:

            capex_result = {
                "value":
                    abs(
                        value
                    ),

                "source":
                    "Yahoo Finance exact-period fallback",

                "tag":
                    None,
            }

    operating_cash_flow = (
        ocf_result[
            "value"
        ]
        if ocf_result
        else None
    )

    capital_expenditures = (
        capex_result[
            "value"
        ]
        if capex_result
        else None
    )

    if capital_expenditures is not None:

        capital_expenditures = abs(
            capital_expenditures
        )

    free_cash_flow = None

    if (
        operating_cash_flow is not None
        and capital_expenditures is not None
    ):

        free_cash_flow = (
            operating_cash_flow
            - capital_expenditures
        )

    cash_fact = find_point_fact(
        company_facts,
        CASH_TAGS,
        target_end,
        preferred_units=[
            "USD"
        ],
    )

    cash = (
        cash_fact[
            "value"
        ]
        if cash_fact
        else None
    )

    if cash is None:

        cash = yahoo_balance_fallback(
            ticker_obj,
            target_end,
            [
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Cash Equivalents",
            ],
        )

    current_assets_fact = find_point_fact(
        company_facts,
        CURRENT_ASSETS_TAGS,
        target_end,
        preferred_units=[
            "USD"
        ],
    )

    current_assets = (
        current_assets_fact[
            "value"
        ]
        if current_assets_fact
        else None
    )

    if current_assets is None:

        current_assets = yahoo_balance_fallback(
            ticker_obj,
            target_end,
            [
                "Current Assets",
                "Total Current Assets",
            ],
        )

    current_liabilities_fact = find_point_fact(
        company_facts,
        CURRENT_LIABILITIES_TAGS,
        target_end,
        preferred_units=[
            "USD"
        ],
    )

    current_liabilities = (
        current_liabilities_fact[
            "value"
        ]
        if current_liabilities_fact
        else None
    )

    if current_liabilities is None:

        current_liabilities = yahoo_balance_fallback(
            ticker_obj,
            target_end,
            [
                "Current Liabilities",
                "Total Current Liabilities",
            ],
        )

    equity_fact = find_point_fact(
        company_facts,
        EQUITY_TAGS,
        target_end,
        preferred_units=[
            "USD"
        ],
    )

    equity = (
        equity_fact[
            "value"
        ]
        if equity_fact
        else None
    )

    if equity is None:

        equity = yahoo_balance_fallback(
            ticker_obj,
            target_end,
            [
                "Stockholders Equity",
                "Common Stock Equity",
            ],
        )

    debt, debt_source = resolve_total_debt(
        company_facts,
        ticker_obj,
        target_end,
    )

    return {
        "revenue":
            (
                revenue_result[
                    "value"
                ]
                if revenue_result
                else None
            ),

        "gross_profit":
            (
                gross_profit_result[
                    "value"
                ]
                if gross_profit_result
                else None
            ),

        "operating_income":
            (
                operating_income_result[
                    "value"
                ]
                if operating_income_result
                else None
            ),

        "pretax_income":
            (
                pretax_result[
                    "value"
                ]
                if pretax_result
                else None
            ),

        "net_income":
            (
                net_income_result[
                    "value"
                ]
                if net_income_result
                else None
            ),

        "diluted_eps":
            diluted_eps,

        "operating_cash_flow":
            operating_cash_flow,

        "capital_expenditures":
            capital_expenditures,

        "free_cash_flow":
            free_cash_flow,

        "cash":
            cash,

        "debt":
            debt,

        "equity":
            equity,

        "current_assets":
            current_assets,

        "current_liabilities":
            current_liabilities,

        "sources":
            {
                "revenue":
                    (
                        revenue_result[
                            "source"
                        ]
                        if revenue_result
                        else "Unavailable"
                    ),

                "gross_profit":
                    (
                        gross_profit_result[
                            "source"
                        ]
                        if gross_profit_result
                        else "Unavailable"
                    ),

                "operating_income":
                    (
                        operating_income_result[
                            "source"
                        ]
                        if operating_income_result
                        else "Unavailable"
                    ),

                "net_income":
                    (
                        net_income_result[
                            "source"
                        ]
                        if net_income_result
                        else "Unavailable"
                    ),

                "diluted_eps":
                    eps_source
                    or "Unavailable",

                "debt":
                    debt_source,
            },
    }


# =========================================================
# COMPANY METRICS
# =========================================================

def get_company_metrics(
    ticker_symbol
):
    ticker_symbol = (
        ticker_symbol
        .strip()
        .upper()
    )

    print(
        f"Downloading {ticker_symbol}..."
    )

    company_facts = get_company_facts(
        ticker_symbol
    )

    latest_period = get_latest_reporting_period(
        ticker_symbol
    )

    ticker_obj = yf.Ticker(
        ticker_symbol
    )

    try:

        info = ticker_obj.info

    except Exception:

        info = {}

    latest = resolve_period_metrics(
        company_facts,
        ticker_obj,
        latest_period,
    )

    if latest[
        "revenue"
    ] is None:

        raise ValueError(
            "No usable latest-quarter revenue found."
        )

    prior_period = find_prior_year_reporting_period(
        ticker_symbol,
        latest_period[
            "report_date"
        ],
        latest_period[
            "form"
        ],
    )

    prior = None

    if prior_period is not None:

        prior = resolve_period_metrics(
            company_facts,
            ticker_obj,
            prior_period,
        )

    revenue_growth_yoy = growth(
        latest[
            "revenue"
        ],
        (
            prior.get(
                "revenue"
            )
            if prior
            else None
        ),
    )

    operating_income_growth_yoy = growth(
        latest[
            "operating_income"
        ],
        (
            prior.get(
                "operating_income"
            )
            if prior
            else None
        ),
    )

    net_income_growth_yoy = growth(
        latest[
            "net_income"
        ],
        (
            prior.get(
                "net_income"
            )
            if prior
            else None
        ),
    )

    eps_growth_yoy = growth(
        latest[
            "diluted_eps"
        ],
        (
            prior.get(
                "diluted_eps"
            )
            if prior
            else None
        ),
    )

    gross_margin = safe_divide(
        latest[
            "gross_profit"
        ],
        latest[
            "revenue"
        ],
    )

    operating_margin = safe_divide(
        latest[
            "operating_income"
        ],
        latest[
            "revenue"
        ],
    )

    net_margin = safe_divide(
        latest[
            "net_income"
        ],
        latest[
            "revenue"
        ],
    )

    fcf_margin = safe_divide(
        latest[
            "free_cash_flow"
        ],
        latest[
            "revenue"
        ],
    )

    debt_to_equity = safe_divide(
        latest[
            "debt"
        ],
        latest[
            "equity"
        ],
    )

    current_ratio = safe_divide(
        latest[
            "current_assets"
        ],
        latest[
            "current_liabilities"
        ],
    )

    non_operating_gap = None
    non_operating_ratio = None
    unusual_non_operating = False

    if (
        latest[
            "pretax_income"
        ] is not None
        and latest[
            "operating_income"
        ] is not None
    ):

        non_operating_gap = (
            latest[
                "pretax_income"
            ]
            - latest[
                "operating_income"
            ]
        )

        non_operating_ratio = safe_divide(
            non_operating_gap,
            abs(
                latest[
                    "operating_income"
                ]
            ),
        )

        if (
            non_operating_ratio is not None
            and abs(
                non_operating_ratio
            )
            >= 0.25
        ):

            unusual_non_operating = True

    eps_comparability_warning = False

    if (
        eps_growth_yoy is not None
        and net_income_growth_yoy is not None
        and abs(
            eps_growth_yoy
            - net_income_growth_yoy
        )
        >= 0.50
    ):

        eps_comparability_warning = True

    missing_metrics = []

    core_metrics = {
        "revenue_growth_yoy":
            revenue_growth_yoy,

        "operating_income_growth_yoy":
            operating_income_growth_yoy,

        "net_income_growth_yoy":
            net_income_growth_yoy,

        "gross_margin":
            gross_margin,

        "operating_margin":
            operating_margin,

        "net_margin":
            net_margin,

        "free_cash_flow_margin":
            fcf_margin,

        "debt_to_equity":
            debt_to_equity,

        "current_ratio":
            current_ratio,
    }

    for name, value in core_metrics.items():

        if value is None:

            missing_metrics.append(
                name
            )

    if len(
        missing_metrics
    ) == 0:

        data_quality = "High"

    elif len(
        missing_metrics
    ) <= 2:

        data_quality = "Moderate"

    else:

        data_quality = "Limited"

    return {
        "ticker":
            ticker_symbol,

        "company_name":
            info.get(
                "longName",
                company_facts.get(
                    "entityName",
                    ticker_symbol,
                ),
            ),

        "sector":
            info.get(
                "sector"
            ),

        "industry":
            info.get(
                "industry"
            ),

        "period_end":
            latest_period[
                "report_date"
            ],

        "filing_form":
            latest_period[
                "form"
            ],

        "filing_date":
            latest_period[
                "filing_date"
            ],

        "revenue":
            latest[
                "revenue"
            ],

        "gross_profit":
            latest[
                "gross_profit"
            ],

        "operating_income":
            latest[
                "operating_income"
            ],

        "pretax_income":
            latest[
                "pretax_income"
            ],

        "net_income":
            latest[
                "net_income"
            ],

        "diluted_eps":
            latest[
                "diluted_eps"
            ],

        "operating_cash_flow":
            latest[
                "operating_cash_flow"
            ],

        "capital_expenditures":
            latest[
                "capital_expenditures"
            ],

        "free_cash_flow":
            latest[
                "free_cash_flow"
            ],

        "cash":
            latest[
                "cash"
            ],

        "debt":
            latest[
                "debt"
            ],

        "equity":
            latest[
                "equity"
            ],

        "current_assets":
            latest[
                "current_assets"
            ],

        "current_liabilities":
            latest[
                "current_liabilities"
            ],

        "revenue_growth_yoy":
            revenue_growth_yoy,

        "operating_income_growth_yoy":
            operating_income_growth_yoy,

        "net_income_growth_yoy":
            net_income_growth_yoy,

        "eps_growth_yoy":
            eps_growth_yoy,

        "gross_margin":
            gross_margin,

        "operating_margin":
            operating_margin,

        "net_margin":
            net_margin,

        "fcf_margin":
            fcf_margin,

        "debt_to_equity":
            debt_to_equity,

        "current_ratio":
            current_ratio,

        "market_cap":
            info.get(
                "marketCap"
            ),

        "trailing_pe":
            info.get(
                "trailingPE"
            ),

        "forward_pe":
            info.get(
                "forwardPE"
            ),

        "price_to_sales":
            info.get(
                "priceToSalesTrailing12Months"
            ),

        "price_to_book":
            info.get(
                "priceToBook"
            ),

        "enterprise_to_ebitda":
            info.get(
                "enterpriseToEbitda"
            ),

        "enterprise_to_revenue":
            info.get(
                "enterpriseToRevenue"
            ),

        "non_operating_gap":
            non_operating_gap,

        "non_operating_ratio":
            non_operating_ratio,

        "unusual_non_operating":
            unusual_non_operating,

        "eps_comparability_warning":
            eps_comparability_warning,

        "data_quality":
            data_quality,

        "missing_metrics":
            missing_metrics,

        "metric_sources":
            latest[
                "sources"
            ],

        "financial_statement_source":
            (
                "SEC Company Facts with "
                "exact-period Yahoo fallbacks"
            ),

        "valuation_source":
            "Yahoo Finance",
    }


# =========================================================
# MULTI-COMPANY COMPARISON
# =========================================================

def compare_companies(
    tickers,
    max_companies=MAX_COMPANIES,
):
    cleaned = []

    for ticker in tickers:

        ticker = (
            ticker
            .strip()
            .upper()
        )

        if (
            ticker
            and ticker not in cleaned
        ):

            cleaned.append(
                ticker
            )

    cleaned = cleaned[
        :max_companies
    ]

    results = []

    for ticker in cleaned:

        try:

            results.append(
                get_company_metrics(
                    ticker
                )
            )

        except Exception as error:

            print(
                f"Could not load {ticker}: {error}"
            )

    return results


# =========================================================
# TERMINAL DISPLAY
# =========================================================

def main():
    raw = input(
        "Enter up to 10 tickers separated by spaces: "
    )

    tickers = [
        ticker
        for ticker in raw.split()
        if ticker.strip()
    ]

    if len(
        tickers
    ) > MAX_COMPANIES:

        print(
            "\nOnly the first 10 unique tickers will be analyzed.\n"
        )

    companies = compare_companies(
        tickers
    )

    if not companies:

        print(
            "No company data loaded."
        )

        return

    dataframe = pd.DataFrame(
        companies
    )

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        300,
    )

    columns = [
        "ticker",
        "company_name",
        "period_end",
        "filing_form",
        "data_quality",

        "revenue_growth_yoy",
        "operating_income_growth_yoy",
        "net_income_growth_yoy",
        "eps_growth_yoy",

        "gross_margin",
        "operating_margin",
        "net_margin",
        "fcf_margin",

        "debt_to_equity",
        "current_ratio",

        "trailing_pe",
        "forward_pe",
        "price_to_sales",
        "enterprise_to_ebitda",

        "unusual_non_operating",
        "eps_comparability_warning",
    ]

    print(
        "\nCompany Comparison Engine V3:\n"
    )

    print(
        dataframe[
            columns
        ]
    )

    print(
        "\nData-quality notes:\n"
    )

    for company in companies:

        ticker = company[
            "ticker"
        ]

        print(
            f"{ticker}: "
            f"period={company['period_end']} "
            f"form={company['filing_form']} "
            f"quality={company['data_quality']}"
        )

        if company[
            "missing_metrics"
        ]:

            print(
                "  Missing: "
                + ", ".join(
                    company[
                        "missing_metrics"
                    ]
                )
            )

        if company[
            "unusual_non_operating"
        ]:

            ratio = company.get(
                "non_operating_ratio"
            )

            if ratio is not None:

                print(
                    "  ⚠ Large non-operating contribution: "
                    f"{ratio * 100:,.1f}% "
                    "of operating income."
                )

        if company[
            "eps_comparability_warning"
        ]:

            print(
                "  ⚠ EPS growth differs materially from "
                "net-income growth; review comparability."
            )


if __name__ == "__main__":
    main()