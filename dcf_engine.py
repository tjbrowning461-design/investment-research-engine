import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv


# =========================================================
# SETTINGS
# =========================================================

load_dotenv()

FORECAST_YEARS = 5

SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "DCF Valuation Engine contact@example.com",
)

SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}

# These are explicit model assumptions.
# They will later be editable in the dashboard.

DEFAULT_RISK_FREE_RATE = 0.0425
DEFAULT_EQUITY_RISK_PREMIUM = 0.0500
DEFAULT_PRETAX_COST_OF_DEBT = 0.0500

MIN_WACC = 0.060
MAX_WACC = 0.150

MIN_TERMINAL_GROWTH = 0.010
MAX_TERMINAL_GROWTH = 0.040


# =========================================================
# SEC TAGS
# =========================================================

REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "Revenues",
    "SalesRevenueGoodsNet",
]

OPERATING_INCOME_TAGS = [
    "OperatingIncomeLoss",
]

PRETAX_INCOME_TAGS = [
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
]

TAX_TAGS = [
    "IncomeTaxExpenseBenefit",
]

D_AND_A_TAGS = [
    "DepreciationDepletionAndAmortization",
    "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
    "Depreciation",
]

CAPEX_TAGS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsForAdditionsToPropertyPlantAndEquipment",
]

CASH_TAGS = [
    "CashAndCashEquivalentsAtCarryingValue",
]

CURRENT_ASSET_TAGS = [
    "AssetsCurrent",
]

CURRENT_LIABILITY_TAGS = [
    "LiabilitiesCurrent",
]

DEBT_CURRENT_TAGS = [
    "ShortTermBorrowings",
    "LongTermDebtCurrent",
    "ShortTermDebtCurrent",
]

DEBT_LONG_TERM_TAGS = [
    "LongTermDebtNoncurrent",
    "LongTermDebt",
]

INTEREST_EXPENSE_TAGS = [
    "InterestExpenseNonOperating",
    "InterestExpenseDebt",
]

SHARES_TAGS = [
    "CommonStockSharesOutstanding",
]


# =========================================================
# BASIC HELPERS
# =========================================================

def safe_divide(numerator, denominator):

    if numerator is None:
        return None

    if denominator in (
        None,
        0,
    ):
        return None

    return numerator / denominator


def to_float(value):

    if value is None:
        return None

    try:
        return float(value)

    except Exception:
        return None


def clamp(
    value,
    minimum,
    maximum,
):

    if value is None:
        return None

    return max(
        minimum,
        min(
            value,
            maximum,
        ),
    )


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

    if (
        start is None
        or end is None
    ):
        return None

    return (
        end - start
    ).days


def format_money(value):

    if (
        value is None
        or pd.isna(value)
    ):
        return "N/A"

    absolute = abs(
        value
    )

    if absolute >= 1_000_000_000_000:

        return (
            f"${value / 1_000_000_000_000:,.2f}T"
        )

    if absolute >= 1_000_000_000:

        return (
            f"${value / 1_000_000_000:,.2f}B"
        )

    if absolute >= 1_000_000:

        return (
            f"${value / 1_000_000:,.2f}M"
        )

    return f"${value:,.2f}"


def format_percent(value):

    if (
        value is None
        or pd.isna(value)
    ):
        return "N/A"

    return (
        f"{value * 100:,.2f}%"
    )


# =========================================================
# SEC DOWNLOAD
# =========================================================

_CIK_CACHE = None
_FACTS_CACHE = {}


def sec_get_json(url):

    response = requests.get(
        url,
        headers=SEC_HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def get_cik(ticker):

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

    if ticker not in _FACTS_CACHE:

        cik = get_cik(
            ticker
        )

        url = (
            "https://data.sec.gov/api/xbrl/companyfacts/"
            f"CIK{cik}.json"
        )

        _FACTS_CACHE[
            ticker
        ] = sec_get_json(
            url
        )

    return _FACTS_CACHE[
        ticker
    ]


# =========================================================
# SEC FACT HELPERS
# =========================================================

def all_fact_entries(
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

    results = []

    for priority, tag in enumerate(
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

        ordered_units = []

        if preferred_units:

            for unit in preferred_units:

                if unit in units:
                    ordered_units.append(
                        unit
                    )

        for unit in units:

            if unit not in ordered_units:
                ordered_units.append(
                    unit
                )

        for unit in ordered_units:

            for entry in units.get(
                unit,
                [],
            ):

                results.append(
                    {
                        "tag":
                            tag,

                        "unit":
                            unit,

                        "priority":
                            priority,

                        "entry":
                            entry,
                    }
                )

    return results


def choose_best_fact(
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

            -item[
                "priority"
            ],
        ),
        reverse=True,
    )

    return candidates[
        0
    ]


# =========================================================
# ANNUAL SEC HISTORY
# =========================================================

def annual_flow_history(
    company_facts,
    tags,
    preferred_units=None,
    max_years=8,
):

    entries = all_fact_entries(
        company_facts,
        tags,
        preferred_units,
    )

    by_end_date = {}

    for item in entries:

        entry = item[
            "entry"
        ]

        if entry.get(
            "form"
        ) != "10-K":
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

        value = to_float(
            entry.get(
                "val"
            )
        )

        if value is None:
            continue

        end = entry.get(
            "end"
        )

        current = by_end_date.get(
            end
        )

        if current is None:

            by_end_date[
                end
            ] = item

            continue

        old_filed = (
            current[
                "entry"
            ].get(
                "filed",
                "",
            )
        )

        new_filed = entry.get(
            "filed",
            "",
        )

        if (
            new_filed > old_filed
            or (
                new_filed
                == old_filed
                and item[
                    "priority"
                ]
                < current[
                    "priority"
                ]
            )
        ):

            by_end_date[
                end
            ] = item

    history = []

    for end, item in by_end_date.items():

        value = to_float(
            item[
                "entry"
            ].get(
                "val"
            )
        )

        history.append(
            {
                "date":
                    end,

                "value":
                    value,

                "tag":
                    item[
                        "tag"
                    ],
            }
        )

    history.sort(
        key=lambda item: item[
            "date"
        ]
    )

    return history[
        -max_years:
    ]


def annual_point_history(
    company_facts,
    tags,
    preferred_units=None,
    max_years=8,
):

    entries = all_fact_entries(
        company_facts,
        tags,
        preferred_units,
    )

    by_end_date = {}

    for item in entries:

        entry = item[
            "entry"
        ]

        if entry.get(
            "form"
        ) != "10-K":
            continue

        if entry.get(
            "start"
        ):
            continue

        value = to_float(
            entry.get(
                "val"
            )
        )

        if value is None:
            continue

        end = entry.get(
            "end"
        )

        current = by_end_date.get(
            end
        )

        if current is None:

            by_end_date[
                end
            ] = item

            continue

        if (
            entry.get(
                "filed",
                "",
            )
            >= current[
                "entry"
            ].get(
                "filed",
                "",
            )
        ):

            by_end_date[
                end
            ] = item

    history = []

    for end, item in by_end_date.items():

        history.append(
            {
                "date":
                    end,

                "value":
                    to_float(
                        item[
                            "entry"
                        ].get(
                            "val"
                        )
                    ),
            }
        )

    history.sort(
        key=lambda item: item[
            "date"
        ]
    )

    return history[
        -max_years:
    ]


# =========================================================
# HISTORY CALCULATIONS
# =========================================================

def history_to_dict(
    history
):

    return {
        item[
            "date"
        ]:
        item[
            "value"
        ]
        for item
        in history
    }


def growth_rates(
    history
):

    rates = []

    for index in range(
        1,
        len(
            history
        ),
    ):

        previous = history[
            index - 1
        ][
            "value"
        ]

        current = history[
            index
        ][
            "value"
        ]

        if previous in (
            None,
            0,
        ):
            continue

        rates.append(
            (
                current
                - previous
            )
            / abs(
                previous
            )
        )

    return rates


def calculate_cagr(
    history
):

    if len(
        history
    ) < 2:
        return None

    first = history[
        0
    ][
        "value"
    ]

    last = history[
        -1
    ][
        "value"
    ]

    years = (
        len(
            history
        )
        - 1
    )

    if (
        first is None
        or last is None
        or first <= 0
        or last <= 0
        or years <= 0
    ):
        return None

    return (
        (
            last
            / first
        )
        ** (
            1
            / years
        )
        - 1
    )


def weighted_average_recent(
    values
):

    if not values:
        return None

    weights = np.arange(
        1,
        len(
            values
        )
        + 1,
        dtype=float,
    )

    values_array = np.array(
        values,
        dtype=float,
    )

    return float(
        np.average(
            values_array,
            weights=weights,
        )
    )


def median_or_none(
    values
):

    clean = [
        value
        for value
        in values
        if value is not None
        and not pd.isna(
            value
        )
    ]

    if not clean:
        return None

    return float(
        np.median(
            clean
        )
    )


# =========================================================
# YAHOO TTM / MARKET DATA
# =========================================================

def get_statement_row(
    statement,
    names,
):

    if (
        statement is None
        or statement.empty
    ):
        return None

    for name in names:

        if name in statement.index:
            return statement.loc[
                name
            ]

    return None


def recent_values(
    row
):

    if row is None:
        return []

    values = []

    for value in row:

        if pd.isna(
            value
        ):
            continue

        numeric = to_float(
            value
        )

        if numeric is not None:
            values.append(
                numeric
            )

    return values


def sum_quarters(
    statement,
    names,
    count=4,
):

    row = get_statement_row(
        statement,
        names,
    )

    values = recent_values(
        row
    )

    if len(
        values
    ) < count:
        return None

    return sum(
        values[
            :count
        ]
    )


def latest_balance_value(
    statement,
    names,
):

    row = get_statement_row(
        statement,
        names,
    )

    values = recent_values(
        row
    )

    if not values:
        return None

    return values[
        0
    ]


# =========================================================
# BUILD HISTORICAL MODEL
# =========================================================

def build_sec_history(
    company_facts
):

    revenue_history = annual_flow_history(
        company_facts,
        REVENUE_TAGS,
        preferred_units=[
            "USD"
        ],
    )

    operating_income_history = annual_flow_history(
        company_facts,
        OPERATING_INCOME_TAGS,
        preferred_units=[
            "USD"
        ],
    )

    d_and_a_history = annual_flow_history(
        company_facts,
        D_AND_A_TAGS,
        preferred_units=[
            "USD"
        ],
    )

    capex_history = annual_flow_history(
        company_facts,
        CAPEX_TAGS,
        preferred_units=[
            "USD"
        ],
    )

    current_assets_history = annual_point_history(
        company_facts,
        CURRENT_ASSET_TAGS,
        preferred_units=[
            "USD"
        ],
    )

    current_liabilities_history = annual_point_history(
        company_facts,
        CURRENT_LIABILITY_TAGS,
        preferred_units=[
            "USD"
        ],
    )

    cash_history = annual_point_history(
        company_facts,
        CASH_TAGS,
        preferred_units=[
            "USD"
        ],
    )

    revenue_by_date = history_to_dict(
        revenue_history
    )

    op_income_by_date = history_to_dict(
        operating_income_history
    )

    d_and_a_by_date = history_to_dict(
        d_and_a_history
    )

    capex_by_date = history_to_dict(
        capex_history
    )

    current_assets_by_date = history_to_dict(
        current_assets_history
    )

    current_liabilities_by_date = history_to_dict(
        current_liabilities_history
    )

    cash_by_date = history_to_dict(
        cash_history
    )


    # -----------------------------------------------------
    # HISTORICAL MARGINS
    # -----------------------------------------------------

    operating_margins = []

    d_and_a_ratios = []

    capex_ratios = []

    operating_nwc_history = []


    for date, revenue in revenue_by_date.items():

        if (
            revenue is None
            or revenue == 0
        ):
            continue

        if date in op_income_by_date:

            operating_margins.append(
                safe_divide(
                    op_income_by_date[
                        date
                    ],
                    revenue,
                )
            )

        if date in d_and_a_by_date:

            d_and_a_ratios.append(
                safe_divide(
                    abs(
                        d_and_a_by_date[
                            date
                        ]
                    ),
                    revenue,
                )
            )

        if date in capex_by_date:

            capex_ratios.append(
                safe_divide(
                    abs(
                        capex_by_date[
                            date
                        ]
                    ),
                    revenue,
                )
            )

        if (
            date in current_assets_by_date
            and date
            in current_liabilities_by_date
        ):

            current_assets = (
                current_assets_by_date[
                    date
                ]
            )

            current_liabilities = (
                current_liabilities_by_date[
                    date
                ]
            )

            cash = cash_by_date.get(
                date,
                0,
            )

            operating_nwc = (
                current_assets
                - current_liabilities
                - (
                    cash
                    or 0
                )
            )

            operating_nwc_history.append(
                {
                    "date":
                        date,

                    "value":
                        operating_nwc,

                    "revenue":
                        revenue,
                }
            )


    # -----------------------------------------------------
    # WORKING CAPITAL INVESTMENT
    # -----------------------------------------------------

    nwc_investment_ratios = []

    for index in range(
        1,
        len(
            operating_nwc_history
        ),
    ):

        current = (
            operating_nwc_history[
                index
            ]
        )

        previous = (
            operating_nwc_history[
                index - 1
            ]
        )

        change_nwc = (
            current[
                "value"
            ]
            - previous[
                "value"
            ]
        )

        current_revenue = current[
            "revenue"
        ]

        if current_revenue:

            ratio = (
                change_nwc
                / current_revenue
            )

            ratio = clamp(
                ratio,
                -0.08,
                0.08,
            )

            nwc_investment_ratios.append(
                ratio
            )


    return {
        "revenue_history":
            revenue_history,

        "revenue_growth_rates":
            growth_rates(
                revenue_history
            ),

        "revenue_cagr":
            calculate_cagr(
                revenue_history
            ),

        "operating_margins":
            operating_margins,

        "d_and_a_ratios":
            d_and_a_ratios,

        "capex_ratios":
            capex_ratios,

        "nwc_investment_ratios":
            nwc_investment_ratios,
    }


# =========================================================
# FINANCIAL BASE
# =========================================================

def get_financial_base(
    ticker_symbol
):

    ticker_symbol = (
        ticker_symbol
        .strip()
        .upper()
    )

    print(
        f"\nDownloading {ticker_symbol} financial data..."
    )

    company_facts = get_company_facts(
        ticker_symbol
    )

    historical = build_sec_history(
        company_facts
    )

    ticker = yf.Ticker(
        ticker_symbol
    )

    try:

        info = ticker.info

    except Exception:

        info = {}

    quarterly_income = (
        ticker.quarterly_income_stmt
    )

    quarterly_cashflow = (
        ticker.quarterly_cashflow
    )

    quarterly_balance = (
        ticker.quarterly_balance_sheet
    )


    # -----------------------------------------------------
    # TTM OPERATING BASE
    # -----------------------------------------------------

    revenue = sum_quarters(
        quarterly_income,
        [
            "Total Revenue",
            "Operating Revenue",
        ],
    )

    operating_income = sum_quarters(
        quarterly_income,
        [
            "Operating Income",
        ],
    )

    pretax_income = sum_quarters(
        quarterly_income,
        [
            "Pretax Income",
            "Pre Tax Income",
        ],
    )

    tax_expense = sum_quarters(
        quarterly_income,
        [
            "Tax Provision",
            "Income Tax Expense",
        ],
    )

    depreciation = sum_quarters(
        quarterly_cashflow,
        [
            "Depreciation And Amortization",
            "Depreciation",
        ],
    )

    capex = sum_quarters(
        quarterly_cashflow,
        [
            "Capital Expenditure",
            "Capital Expenditures",
        ],
    )

    if capex is not None:

        capex = abs(
            capex
        )

    if revenue is None:

        raise ValueError(
            "Could not determine TTM revenue."
        )

    if operating_income is None:

        raise ValueError(
            "Could not determine TTM operating income."
        )


    # -----------------------------------------------------
    # BALANCE SHEET
    # -----------------------------------------------------

    cash = latest_balance_value(
        quarterly_balance,
        [
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents",
        ],
    )

    debt = latest_balance_value(
        quarterly_balance,
        [
            "Total Debt",
        ],
    )

    cash = (
        cash
        if cash is not None
        else 0
    )

    debt = (
        debt
        if debt is not None
        else 0
    )


    # -----------------------------------------------------
    # PRICE / SHARES
    # -----------------------------------------------------

    current_price = (
        info.get(
            "currentPrice"
        )
        or info.get(
            "regularMarketPrice"
        )
    )

    market_cap = info.get(
        "marketCap"
    )

    shares_outstanding = info.get(
        "sharesOutstanding"
    )

    if (
        shares_outstanding is None
        and market_cap is not None
        and current_price not in (
            None,
            0,
        )
    ):

        shares_outstanding = (
            market_cap
            / current_price
        )

    if shares_outstanding is None:

        raise ValueError(
            "Could not determine shares outstanding."
        )


    # -----------------------------------------------------
    # CURRENT RATIOS
    # -----------------------------------------------------

    operating_margin = safe_divide(
        operating_income,
        revenue,
    )

    effective_tax_rate = safe_divide(
        tax_expense,
        pretax_income,
    )

    if (
        effective_tax_rate is None
        or effective_tax_rate < 0.05
        or effective_tax_rate > 0.35
    ):

        effective_tax_rate = 0.21


    # -----------------------------------------------------
    # NORMALIZED D&A / CAPEX
    # -----------------------------------------------------

    current_d_and_a_ratio = safe_divide(
        depreciation,
        revenue,
    )

    current_capex_ratio = safe_divide(
        capex,
        revenue,
    )

    normalized_d_and_a_ratio = median_or_none(
        historical[
            "d_and_a_ratios"
        ]
    )

    normalized_capex_ratio = median_or_none(
        historical[
            "capex_ratios"
        ]
    )

    if normalized_d_and_a_ratio is None:

        normalized_d_and_a_ratio = (
            current_d_and_a_ratio
            if current_d_and_a_ratio is not None
            else 0.03
        )

    if normalized_capex_ratio is None:

        normalized_capex_ratio = (
            current_capex_ratio
            if current_capex_ratio is not None
            else 0.03
        )


    # -----------------------------------------------------
    # NORMALIZED WORKING CAPITAL
    # -----------------------------------------------------

    normalized_nwc_investment_ratio = (
        median_or_none(
            historical[
                "nwc_investment_ratios"
            ]
        )
    )

    if normalized_nwc_investment_ratio is None:

        normalized_nwc_investment_ratio = 0.01

    normalized_nwc_investment_ratio = clamp(
        normalized_nwc_investment_ratio,
        -0.05,
        0.05,
    )


    # -----------------------------------------------------
    # NORMALIZED OPERATING MARGIN
    # -----------------------------------------------------

    normalized_margin = median_or_none(
        historical[
            "operating_margins"
        ]
    )

    if normalized_margin is None:

        normalized_margin = operating_margin


    # -----------------------------------------------------
    # GROWTH ANALYSIS
    # -----------------------------------------------------

    growth_rates_history = historical[
        "revenue_growth_rates"
    ]

    historical_cagr = historical[
        "revenue_cagr"
    ]

    weighted_recent_growth = (
        weighted_average_recent(
            growth_rates_history[
                -5:
            ]
        )
    )

    if weighted_recent_growth is None:

        weighted_recent_growth = 0.06

    if historical_cagr is None:

        historical_cagr = (
            weighted_recent_growth
        )


    # -----------------------------------------------------
    # WACC
    # -----------------------------------------------------

    beta = info.get(
        "beta"
    )

    if beta is None:

        beta = 1.0

    cost_of_equity = (
        DEFAULT_RISK_FREE_RATE
        + beta
        * DEFAULT_EQUITY_RISK_PREMIUM
    )


    # Estimate debt cost from reported interest expense where possible.

    annual_interest_history = annual_flow_history(
        company_facts,
        INTEREST_EXPENSE_TAGS,
        preferred_units=[
            "USD"
        ],
    )

    latest_interest_expense = None

    if annual_interest_history:

        latest_interest_expense = abs(
            annual_interest_history[
                -1
            ][
                "value"
            ]
        )

    estimated_pretax_cost_of_debt = (
        safe_divide(
            latest_interest_expense,
            debt,
        )
    )

    if (
        estimated_pretax_cost_of_debt is None
        or estimated_pretax_cost_of_debt <= 0
        or estimated_pretax_cost_of_debt > 0.15
    ):

        estimated_pretax_cost_of_debt = (
            DEFAULT_PRETAX_COST_OF_DEBT
        )

    after_tax_cost_of_debt = (
        estimated_pretax_cost_of_debt
        * (
            1
            - effective_tax_rate
        )
    )

    total_capital = (
        (
            market_cap
            or 0
        )
        + debt
    )

    if total_capital > 0:

        equity_weight = (
            (
                market_cap
                or 0
            )
            / total_capital
        )

        debt_weight = (
            debt
            / total_capital
        )

        estimated_wacc = (
            equity_weight
            * cost_of_equity
            + debt_weight
            * after_tax_cost_of_debt
        )

    else:

        equity_weight = None
        debt_weight = None
        estimated_wacc = 0.09

    estimated_wacc = clamp(
        estimated_wacc,
        MIN_WACC,
        MAX_WACC,
    )


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

        "revenue":
            revenue,

        "operating_income":
            operating_income,

        "operating_margin":
            operating_margin,

        "normalized_operating_margin":
            normalized_margin,

        "tax_rate":
            effective_tax_rate,

        "depreciation":
            depreciation,

        "d_and_a_percent":
            normalized_d_and_a_ratio,

        "capex":
            capex,

        "capex_percent":
            normalized_capex_ratio,

        "nwc_investment_percent":
            normalized_nwc_investment_ratio,

        "cash":
            cash,

        "debt":
            debt,

        "shares_outstanding":
            shares_outstanding,

        "current_price":
            current_price,

        "market_cap":
            market_cap,

        "beta":
            beta,

        "risk_free_rate":
            DEFAULT_RISK_FREE_RATE,

        "equity_risk_premium":
            DEFAULT_EQUITY_RISK_PREMIUM,

        "cost_of_equity":
            cost_of_equity,

        "pretax_cost_of_debt":
            estimated_pretax_cost_of_debt,

        "after_tax_cost_of_debt":
            after_tax_cost_of_debt,

        "equity_weight":
            equity_weight,

        "debt_weight":
            debt_weight,

        "estimated_wacc":
            estimated_wacc,

        "historical_revenue":
            historical[
                "revenue_history"
            ],

        "revenue_growth_history":
            growth_rates_history,

        "historical_revenue_cagr":
            historical_cagr,

        "weighted_recent_growth":
            weighted_recent_growth,

        "historical_operating_margins":
            historical[
                "operating_margins"
            ],

        "historical_nwc_investment":
            historical[
                "nwc_investment_ratios"
            ],
    }


# =========================================================
# COMPANY-SPECIFIC ASSUMPTIONS
# =========================================================

def build_path(
    start,
    end,
    years=FORECAST_YEARS,
):

    return list(
        np.linspace(
            start,
            end,
            years,
        )
    )


def derive_base_growth(
    financial_base
):

    weighted_recent = financial_base[
        "weighted_recent_growth"
    ]

    cagr = financial_base[
        "historical_revenue_cagr"
    ]

    # More weight on recent performance,
    # while still anchoring to longer history.

    base_growth = (
        0.65
        * weighted_recent
        + 0.35
        * cagr
    )

    return clamp(
        base_growth,
        -0.05,
        0.30,
    )


def build_company_scenarios(
    financial_base
):

    base_growth = derive_base_growth(
        financial_base
    )

    current_margin = financial_base[
        "operating_margin"
    ]

    normalized_margin = financial_base[
        "normalized_operating_margin"
    ]

    base_wacc = financial_base[
        "estimated_wacc"
    ]

    tax_rate = financial_base[
        "tax_rate"
    ]

    d_and_a_percent = financial_base[
        "d_and_a_percent"
    ]

    capex_percent = financial_base[
        "capex_percent"
    ]

    nwc_percent = financial_base[
        "nwc_investment_percent"
    ]


    # -----------------------------------------------------
    # BASE CASE
    # -----------------------------------------------------

    base_terminal_growth = 0.025

    base_end_growth = max(
        base_terminal_growth
        + 0.015,
        0.035,
    )

    base_growth_path = build_path(
        base_growth,
        base_end_growth,
    )

    base_margin_path = build_path(
        current_margin,
        normalized_margin,
    )


    # -----------------------------------------------------
    # BEAR CASE
    # -----------------------------------------------------

    bear_growth_path = build_path(
        max(
            base_growth
            - 0.05,
            -0.05,
        ),
        0.020,
    )

    bear_margin_path = build_path(
        max(
            current_margin
            - 0.02,
            0.01,
        ),
        max(
            normalized_margin
            - 0.03,
            0.01,
        ),
    )

    bear_wacc = clamp(
        base_wacc
        + 0.015,
        MIN_WACC,
        MAX_WACC,
    )


    # -----------------------------------------------------
    # BULL CASE
    # -----------------------------------------------------

    bull_growth_path = build_path(
        min(
            base_growth
            + 0.05,
            0.35,
        ),
        0.060,
    )

    bull_target_margin = min(
        max(
            current_margin,
            normalized_margin,
        )
        + 0.025,
        0.65,
    )

    bull_margin_path = build_path(
        current_margin,
        bull_target_margin,
    )

    bull_wacc = clamp(
        base_wacc
        - 0.010,
        MIN_WACC,
        MAX_WACC,
    )


    return {
        "Bear": {
            "growth_path":
                bear_growth_path,

            "margin_path":
                bear_margin_path,

            "tax_rate":
                tax_rate,

            "d_and_a_percent":
                d_and_a_percent,

            "capex_percent":
                capex_percent,

            "nwc_investment_percent":
                min(
                    nwc_percent
                    + 0.01,
                    0.06,
                ),

            "wacc":
                bear_wacc,

            "terminal_growth":
                0.015,
        },

        "Base": {
            "growth_path":
                base_growth_path,

            "margin_path":
                base_margin_path,

            "tax_rate":
                tax_rate,

            "d_and_a_percent":
                d_and_a_percent,

            "capex_percent":
                capex_percent,

            "nwc_investment_percent":
                nwc_percent,

            "wacc":
                base_wacc,

            "terminal_growth":
                base_terminal_growth,
        },

        "Bull": {
            "growth_path":
                bull_growth_path,

            "margin_path":
                bull_margin_path,

            "tax_rate":
                tax_rate,

            "d_and_a_percent":
                d_and_a_percent,

            "capex_percent":
                capex_percent,

            "nwc_investment_percent":
                max(
                    nwc_percent
                    - 0.005,
                    -0.05,
                ),

            "wacc":
                bull_wacc,

            "terminal_growth":
                0.030,
        },
    }


# =========================================================
# DCF CORE
# =========================================================

def forecast_dcf(
    financial_base,
    growth_path,
    margin_path,
    tax_rate,
    d_and_a_percent,
    capex_percent,
    nwc_investment_percent,
    wacc,
    terminal_growth,
):

    years = len(
        growth_path
    )

    if len(
        margin_path
    ) != years:

        raise ValueError(
            "Growth path and margin path must have the same length."
        )

    if wacc <= terminal_growth:

        raise ValueError(
            "WACC must be greater than terminal growth."
        )

    revenue = financial_base[
        "revenue"
    ]

    forecast = []

    total_pv_fcf = 0


    for year in range(
        1,
        years + 1,
    ):

        growth = growth_path[
            year - 1
        ]

        margin = margin_path[
            year - 1
        ]

        revenue = (
            revenue
            * (
                1
                + growth
            )
        )

        ebit = (
            revenue
            * margin
        )

        cash_taxes = (
            ebit
            * tax_rate
        )

        nopat = (
            ebit
            - cash_taxes
        )

        d_and_a = (
            revenue
            * d_and_a_percent
        )

        capex = (
            revenue
            * capex_percent
        )

        change_nwc = (
            revenue
            * nwc_investment_percent
        )

        free_cash_flow = (
            nopat
            + d_and_a
            - capex
            - change_nwc
        )

        discount_factor = (
            1
            / (
                (
                    1
                    + wacc
                )
                ** year
            )
        )

        pv_fcf = (
            free_cash_flow
            * discount_factor
        )

        total_pv_fcf += (
            pv_fcf
        )

        forecast.append(
            {
                "year":
                    year,

                "revenue_growth":
                    growth,

                "operating_margin":
                    margin,

                "revenue":
                    revenue,

                "ebit":
                    ebit,

                "taxes":
                    cash_taxes,

                "nopat":
                    nopat,

                "d_and_a":
                    d_and_a,

                "capex":
                    capex,

                "change_nwc":
                    change_nwc,

                "free_cash_flow":
                    free_cash_flow,

                "discount_factor":
                    discount_factor,

                "present_value_fcf":
                    pv_fcf,
            }
        )


    # -----------------------------------------------------
    # TERMINAL VALUE
    # -----------------------------------------------------

    final_fcf = forecast[
        -1
    ][
        "free_cash_flow"
    ]

    terminal_fcf = (
        final_fcf
        * (
            1
            + terminal_growth
        )
    )

    terminal_value = (
        terminal_fcf
        / (
            wacc
            - terminal_growth
        )
    )

    terminal_discount_factor = (
        1
        / (
            (
                1
                + wacc
            )
            ** years
        )
    )

    present_value_terminal = (
        terminal_value
        * terminal_discount_factor
    )


    # -----------------------------------------------------
    # VALUE
    # -----------------------------------------------------

    enterprise_value = (
        total_pv_fcf
        + present_value_terminal
    )

    equity_value = (
        enterprise_value
        + financial_base[
            "cash"
        ]
        - financial_base[
            "debt"
        ]
    )

    intrinsic_value_per_share = (
        equity_value
        / financial_base[
            "shares_outstanding"
        ]
    )

    current_price = financial_base[
        "current_price"
    ]

    upside_downside = None

    if current_price not in (
        None,
        0,
    ):

        upside_downside = (
            intrinsic_value_per_share
            / current_price
        ) - 1

    terminal_value_percent = (
        safe_divide(
            present_value_terminal,
            enterprise_value,
        )
    )

    return {
        "forecast":
            forecast,

        "present_value_forecast_fcf":
            total_pv_fcf,

        "terminal_value":
            terminal_value,

        "present_value_terminal":
            present_value_terminal,

        "terminal_value_percent":
            terminal_value_percent,

        "enterprise_value":
            enterprise_value,

        "equity_value":
            equity_value,

        "intrinsic_value_per_share":
            intrinsic_value_per_share,

        "current_price":
            current_price,

        "upside_downside":
            upside_downside,

        "assumptions": {
            "growth_path":
                growth_path,

            "margin_path":
                margin_path,

            "tax_rate":
                tax_rate,

            "d_and_a_percent":
                d_and_a_percent,

            "capex_percent":
                capex_percent,

            "nwc_investment_percent":
                nwc_investment_percent,

            "wacc":
                wacc,

            "terminal_growth":
                terminal_growth,
        },
    }


# =========================================================
# SCENARIOS
# =========================================================

def run_scenarios(
    financial_base
):

    assumptions = (
        build_company_scenarios(
            financial_base
        )
    )

    results = {}

    for name, scenario in assumptions.items():

        results[
            name
        ] = forecast_dcf(
            financial_base,
            **scenario,
        )

    return (
        assumptions,
        results,
    )


# =========================================================
# REVERSE DCF
# =========================================================

def reverse_dcf_growth(
    financial_base,
    base_assumptions,
    target_price=None,
):

    if target_price is None:

        target_price = financial_base[
            "current_price"
        ]

    if target_price in (
        None,
        0,
    ):

        return None

    low_growth = -0.10
    high_growth = 0.50

    ending_growth = max(
        base_assumptions[
            "terminal_growth"
        ]
        + 0.015,
        0.035,
    )

    def value_for_start_growth(
        starting_growth
    ):

        growth_path = build_path(
            starting_growth,
            ending_growth,
        )

        result = forecast_dcf(
            financial_base,
            growth_path=growth_path,
            margin_path=(
                base_assumptions[
                    "margin_path"
                ]
            ),
            tax_rate=(
                base_assumptions[
                    "tax_rate"
                ]
            ),
            d_and_a_percent=(
                base_assumptions[
                    "d_and_a_percent"
                ]
            ),
            capex_percent=(
                base_assumptions[
                    "capex_percent"
                ]
            ),
            nwc_investment_percent=(
                base_assumptions[
                    "nwc_investment_percent"
                ]
            ),
            wacc=(
                base_assumptions[
                    "wacc"
                ]
            ),
            terminal_growth=(
                base_assumptions[
                    "terminal_growth"
                ]
            ),
        )

        return result[
            "intrinsic_value_per_share"
        ]

    low_value = value_for_start_growth(
        low_growth
    )

    high_value = value_for_start_growth(
        high_growth
    )

    if target_price < low_value:

        return {
            "implied_starting_growth":
                low_growth,

            "bounded":
                True,

            "note":
                (
                    "Market price is below the lower "
                    "reverse-DCF search boundary."
                ),
        }

    if target_price > high_value:

        return {
            "implied_starting_growth":
                high_growth,

            "bounded":
                True,

            "note":
                (
                    "Market price requires growth above "
                    "the reverse-DCF search boundary."
                ),
        }

    for _ in range(
        80
    ):

        midpoint = (
            low_growth
            + high_growth
        ) / 2

        midpoint_value = (
            value_for_start_growth(
                midpoint
            )
        )

        if midpoint_value < target_price:

            low_growth = midpoint

        else:

            high_growth = midpoint

    implied_growth = (
        low_growth
        + high_growth
    ) / 2

    return {
        "implied_starting_growth":
            implied_growth,

        "ending_growth":
            ending_growth,

        "bounded":
            False,

        "note":
            (
                "Revenue growth required by the current "
                "market price under base-case margins, "
                "WACC, and terminal growth."
            ),
    }


# =========================================================
# SENSITIVITY TABLE
# =========================================================

def sensitivity_table(
    financial_base,
    base_assumptions,
):

    center_wacc = (
        base_assumptions[
            "wacc"
        ]
    )

    wacc_values = [
        clamp(
            center_wacc
            + adjustment,
            MIN_WACC,
            MAX_WACC,
        )
        for adjustment
        in [
            -0.020,
            -0.010,
            0.000,
            0.010,
            0.020,
        ]
    ]

    center_terminal = (
        base_assumptions[
            "terminal_growth"
        ]
    )

    terminal_values = [
        clamp(
            center_terminal
            + adjustment,
            MIN_TERMINAL_GROWTH,
            MAX_TERMINAL_GROWTH,
        )
        for adjustment
        in [
            -0.010,
            -0.005,
            0.000,
            0.005,
            0.010,
        ]
    ]

    wacc_values = sorted(
        set(
            round(
                value,
                4,
            )
            for value
            in wacc_values
        )
    )

    terminal_values = sorted(
        set(
            round(
                value,
                4,
            )
            for value
            in terminal_values
        )
    )

    table = pd.DataFrame(
        index=[
            f"{value * 100:.1f}%"
            for value
            in terminal_values
        ],
        columns=[
            f"{value * 100:.2f}%"
            for value
            in wacc_values
        ],
    )

    for terminal_growth in terminal_values:

        for wacc in wacc_values:

            if wacc <= terminal_growth:

                table.loc[
                    f"{terminal_growth * 100:.1f}%",
                    f"{wacc * 100:.2f}%",
                ] = np.nan

                continue

            result = forecast_dcf(
                financial_base,
                growth_path=(
                    base_assumptions[
                        "growth_path"
                    ]
                ),
                margin_path=(
                    base_assumptions[
                        "margin_path"
                    ]
                ),
                tax_rate=(
                    base_assumptions[
                        "tax_rate"
                    ]
                ),
                d_and_a_percent=(
                    base_assumptions[
                        "d_and_a_percent"
                    ]
                ),
                capex_percent=(
                    base_assumptions[
                        "capex_percent"
                    ]
                ),
                nwc_investment_percent=(
                    base_assumptions[
                        "nwc_investment_percent"
                    ]
                ),
                wacc=wacc,
                terminal_growth=(
                    terminal_growth
                ),
            )

            table.loc[
                f"{terminal_growth * 100:.1f}%",
                f"{wacc * 100:.2f}%",
            ] = round(
                result[
                    "intrinsic_value_per_share"
                ],
                2,
            )

    table.index.name = (
        "Terminal Growth"
    )

    table.columns.name = (
        "WACC"
    )

    return table


# =========================================================
# MODEL QUALITY
# =========================================================

def classify_stability(
    values,
    low_threshold,
    high_threshold,
):

    if (
        values is None
        or len(
            values
        ) < 2
    ):

        return "Limited"

    standard_deviation = float(
        np.std(
            values
        )
    )

    if standard_deviation <= low_threshold:

        return "High"

    if standard_deviation <= high_threshold:

        return "Moderate"

    return "Low"


def build_model_quality(
    financial_base,
    scenarios,
):

    base = scenarios[
        "Base"
    ]

    terminal_share = base[
        "terminal_value_percent"
    ]

    if terminal_share is None:

        terminal_reliance = (
            "Unknown"
        )

    elif terminal_share < 0.65:

        terminal_reliance = (
            "Low"
        )

    elif terminal_share < 0.80:

        terminal_reliance = (
            "Moderate"
        )

    else:

        terminal_reliance = (
            "High"
        )

    growth_stability = classify_stability(
        financial_base[
            "revenue_growth_history"
        ],
        low_threshold=0.05,
        high_threshold=0.12,
    )

    margin_stability = classify_stability(
        financial_base[
            "historical_operating_margins"
        ],
        low_threshold=0.02,
        high_threshold=0.05,
    )

    annual_years = len(
        financial_base[
            "historical_revenue"
        ]
    )

    if annual_years >= 5:

        data_quality = "High"

    elif annual_years >= 3:

        data_quality = "Moderate"

    else:

        data_quality = "Limited"

    bear_value = scenarios[
        "Bear"
    ][
        "intrinsic_value_per_share"
    ]

    bull_value = scenarios[
        "Bull"
    ][
        "intrinsic_value_per_share"
    ]

    base_value = scenarios[
        "Base"
    ][
        "intrinsic_value_per_share"
    ]

    scenario_spread = None

    if base_value not in (
        None,
        0,
    ):

        scenario_spread = (
            bull_value
            - bear_value
        ) / abs(
            base_value
        )

    if (
        scenario_spread is None
        or scenario_spread > 1.00
    ):

        forecast_uncertainty = "High"

    elif scenario_spread > 0.50:

        forecast_uncertainty = "Moderate"

    else:

        forecast_uncertainty = "Low"


    # -----------------------------------------------------
    # OVERALL CONFIDENCE
    # -----------------------------------------------------

    score = 0

    if data_quality == "High":
        score += 2

    elif data_quality == "Moderate":
        score += 1

    if growth_stability == "High":
        score += 2

    elif growth_stability == "Moderate":
        score += 1

    if margin_stability == "High":
        score += 2

    elif margin_stability == "Moderate":
        score += 1

    if terminal_reliance == "Low":
        score += 2

    elif terminal_reliance == "Moderate":
        score += 1

    if forecast_uncertainty == "Low":
        score += 2

    elif forecast_uncertainty == "Moderate":
        score += 1

    if score >= 8:

        overall_confidence = "High"

    elif score >= 5:

        overall_confidence = "Moderate"

    else:

        overall_confidence = "Low"


    warnings = []

    if terminal_reliance == "High":

        warnings.append(
            (
                "A large portion of enterprise value comes "
                "from terminal value, making the DCF highly "
                "sensitive to long-term assumptions."
            )
        )

    if growth_stability == "Low":

        warnings.append(
            (
                "Historical revenue growth has been volatile."
            )
        )

    if margin_stability == "Low":

        warnings.append(
            (
                "Historical operating margins have been volatile."
            )
        )

    if data_quality != "High":

        warnings.append(
            (
                "Historical SEC coverage is shorter than ideal."
            )
        )

    if forecast_uncertainty == "High":

        warnings.append(
            (
                "Bear-to-bull valuation dispersion is very wide."
            )
        )


    return {
        "data_quality":
            data_quality,

        "historical_growth_stability":
            growth_stability,

        "margin_stability":
            margin_stability,

        "terminal_value_reliance":
            terminal_reliance,

        "forecast_uncertainty":
            forecast_uncertainty,

        "overall_confidence":
            overall_confidence,

        "scenario_spread":
            scenario_spread,

        "warnings":
            warnings,
    }


# =========================================================
# COMPLETE DCF ANALYSIS
# =========================================================

def analyze_dcf(
    ticker_symbol
):

    financial_base = get_financial_base(
        ticker_symbol
    )

    assumptions, scenarios = (
        run_scenarios(
            financial_base
        )
    )

    reverse_dcf = reverse_dcf_growth(
        financial_base,
        assumptions[
            "Base"
        ],
    )

    sensitivity = sensitivity_table(
        financial_base,
        assumptions[
            "Base"
        ],
    )

    model_quality = build_model_quality(
        financial_base,
        scenarios,
    )

    values = [
        scenarios[
            "Bear"
        ][
            "intrinsic_value_per_share"
        ],

        scenarios[
            "Base"
        ][
            "intrinsic_value_per_share"
        ],

        scenarios[
            "Bull"
        ][
            "intrinsic_value_per_share"
        ],
    ]

    valuation_range = {
        "low":
            min(
                values
            ),

        "base":
            scenarios[
                "Base"
            ][
                "intrinsic_value_per_share"
            ],

        "high":
            max(
                values
            ),
    }

    return {
        "financial_base":
            financial_base,

        "assumptions":
            assumptions,

        "scenarios":
            scenarios,

        "reverse_dcf":
            reverse_dcf,

        "sensitivity":
            sensitivity,

        "model_quality":
            model_quality,

        "valuation_range":
            valuation_range,
    }


# =========================================================
# DISPLAY
# =========================================================

def print_financial_base(
    company
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"{company['company_name']} "
        f"({company['ticker']})"
    )

    print(
        "=" * 80
    )

    print(
        f"TTM Revenue:                    "
        f"{format_money(company['revenue'])}"
    )

    print(
        f"TTM Operating Income:           "
        f"{format_money(company['operating_income'])}"
    )

    print(
        f"Current Operating Margin:       "
        f"{format_percent(company['operating_margin'])}"
    )

    print(
        f"Normalized Operating Margin:    "
        f"{format_percent(company['normalized_operating_margin'])}"
    )

    print(
        f"Historical Revenue CAGR:        "
        f"{format_percent(company['historical_revenue_cagr'])}"
    )

    print(
        f"Weighted Recent Growth:         "
        f"{format_percent(company['weighted_recent_growth'])}"
    )

    print(
        f"Tax Rate:                       "
        f"{format_percent(company['tax_rate'])}"
    )

    print(
        f"Normalized D&A / Revenue:       "
        f"{format_percent(company['d_and_a_percent'])}"
    )

    print(
        f"Normalized CapEx / Revenue:     "
        f"{format_percent(company['capex_percent'])}"
    )

    print(
        f"Normalized ΔNWC / Revenue:      "
        f"{format_percent(company['nwc_investment_percent'])}"
    )

    print(
        f"Cash:                           "
        f"{format_money(company['cash'])}"
    )

    print(
        f"Debt:                           "
        f"{format_money(company['debt'])}"
    )

    print(
        f"Shares Outstanding:             "
        f"{company['shares_outstanding'] / 1_000_000_000:,.2f}B"
    )

    print(
        f"Current Price:                  "
        f"{format_money(company['current_price'])}"
    )


def print_cost_of_capital(
    company
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "COST OF CAPITAL"
    )

    print(
        "=" * 80
    )

    print(
        f"Beta:                           "
        f"{company['beta']:,.2f}"
    )

    print(
        f"Risk-Free Rate:                 "
        f"{format_percent(company['risk_free_rate'])}"
    )

    print(
        f"Equity Risk Premium:            "
        f"{format_percent(company['equity_risk_premium'])}"
    )

    print(
        f"Cost of Equity:                 "
        f"{format_percent(company['cost_of_equity'])}"
    )

    print(
        f"Pre-Tax Cost of Debt:           "
        f"{format_percent(company['pretax_cost_of_debt'])}"
    )

    print(
        f"After-Tax Cost of Debt:         "
        f"{format_percent(company['after_tax_cost_of_debt'])}"
    )

    print(
        f"Estimated WACC:                 "
        f"{format_percent(company['estimated_wacc'])}"
    )


def print_scenario(
    name,
    result,
):

    assumptions = result[
        "assumptions"
    ]

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"{name.upper()} CASE"
    )

    print(
        "-" * 80
    )

    print(
        "Revenue Growth Path:  "
        + " → ".join(
            format_percent(
                value
            )
            for value
            in assumptions[
                "growth_path"
            ]
        )
    )

    print(
        "Operating Margin Path:"
    )

    print(
        "  "
        + " → ".join(
            format_percent(
                value
            )
            for value
            in assumptions[
                "margin_path"
            ]
        )
    )

    print(
        f"WACC:                 "
        f"{format_percent(assumptions['wacc'])}"
    )

    print(
        f"Terminal Growth:      "
        f"{format_percent(assumptions['terminal_growth'])}"
    )

    print(
        f"Intrinsic Value:      "
        f"${result['intrinsic_value_per_share']:,.2f}/share"
    )

    print(
        f"Current Price:        "
        f"{format_money(result['current_price'])}"
    )

    print(
        f"Upside / Downside:    "
        f"{format_percent(result['upside_downside'])}"
    )

    print(
        f"Terminal Value % EV:  "
        f"{format_percent(result['terminal_value_percent'])}"
    )


def print_base_forecast(
    result
):

    dataframe = pd.DataFrame(
        result[
            "forecast"
        ]
    )

    display = dataframe[
        [
            "year",
            "revenue_growth",
            "operating_margin",
            "revenue",
            "ebit",
            "nopat",
            "d_and_a",
            "capex",
            "change_nwc",
            "free_cash_flow",
            "present_value_fcf",
        ]
    ].copy()

    for column in [
        "revenue",
        "ebit",
        "nopat",
        "d_and_a",
        "capex",
        "change_nwc",
        "free_cash_flow",
        "present_value_fcf",
    ]:

        display[
            column
        ] = (
            display[
                column
            ]
            / 1_000_000_000
        )

    display[
        "revenue_growth"
    ] *= 100

    display[
        "operating_margin"
    ] *= 100

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "BASE-CASE 5-YEAR FORECAST"
    )

    print(
        "Percent columns shown as percentage points. "
        "Financial values shown in $ billions."
    )

    print(
        "=" * 80
    )

    print(
        display.to_string(
            index=False,
            float_format=lambda value:
                f"{value:,.2f}",
        )
    )


# =========================================================
# MAIN
# =========================================================

def main():

    ticker = input(
        "Enter stock ticker: "
    )

    try:

        analysis = analyze_dcf(
            ticker
        )

    except Exception as error:

        print(
            f"\nCould not build DCF: {error}"
        )

        return

    company = analysis[
        "financial_base"
    ]

    scenarios = analysis[
        "scenarios"
    ]

    quality = analysis[
        "model_quality"
    ]

    reverse = analysis[
        "reverse_dcf"
    ]

    valuation_range = analysis[
        "valuation_range"
    ]


    # -----------------------------------------------------
    # FINANCIAL BASE
    # -----------------------------------------------------

    print_financial_base(
        company
    )

    print_cost_of_capital(
        company
    )


    # -----------------------------------------------------
    # HISTORICAL GROWTH
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "HISTORICAL REVENUE GROWTH"
    )

    print(
        "=" * 80
    )

    for index, growth in enumerate(
        company[
            "revenue_growth_history"
        ],
        start=1,
    ):

        print(
            f"Historical Year {index}: "
            f"{format_percent(growth)}"
        )


    # -----------------------------------------------------
    # DCF
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL COMPANY-SPECIFIC DCF"
    )

    print(
        "=" * 80
    )

    for name in [
        "Bear",
        "Base",
        "Bull",
    ]:

        print_scenario(
            name,
            scenarios[
                name
            ],
        )


    # -----------------------------------------------------
    # RANGE
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "VALUATION RANGE"
    )

    print(
        "=" * 80
    )

    print(
        f"Bear / Low:     "
        f"${valuation_range['low']:,.2f}"
    )

    print(
        f"Base:           "
        f"${valuation_range['base']:,.2f}"
    )

    print(
        f"Bull / High:    "
        f"${valuation_range['high']:,.2f}"
    )

    print(
        f"Current Price:  "
        f"{format_money(company['current_price'])}"
    )


    # -----------------------------------------------------
    # BASE FORECAST
    # -----------------------------------------------------

    print_base_forecast(
        scenarios[
            "Base"
        ]
    )


    # -----------------------------------------------------
    # REVERSE DCF
    # -----------------------------------------------------

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "REVERSE DCF"
    )

    print(
        "=" * 80
    )

    if reverse is None:

        print(
            "Reverse DCF unavailable."
        )

    else:

        print(
            "Market-Implied Starting Revenue Growth: "
            f"{format_percent(reverse['implied_starting_growth'])}"
        )

        if reverse.get(
            "ending_growth"
        ) is not None:

            print(
                "Growth fades toward:                  "
                f"{format_percent(reverse['ending_growth'])}"
            )

        print(
            reverse[
                "note"
            ]
        )


    # -----------------------------------------------------
    # MODEL QUALITY
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MODEL QUALITY"
    )

    print(
        "=" * 80
    )

    print(
        f"Data Quality:              "
        f"{quality['data_quality']}"
    )

    print(
        f"Historical Growth Stability:"
        f" {quality['historical_growth_stability']}"
    )

    print(
        f"Margin Stability:           "
        f"{quality['margin_stability']}"
    )

    print(
        f"Terminal Value Reliance:    "
        f"{quality['terminal_value_reliance']}"
    )

    print(
        f"Forecast Uncertainty:       "
        f"{quality['forecast_uncertainty']}"
    )

    print(
        f"Overall DCF Confidence:     "
        f"{quality['overall_confidence']}"
    )

    if quality[
        "warnings"
    ]:

        print(
            "\nWarnings:"
        )

        for warning in quality[
            "warnings"
        ]:

            print(
                f"• {warning}"
            )


    # -----------------------------------------------------
    # SENSITIVITY
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DCF SENSITIVITY TABLE"
    )

    print(
        "Estimated intrinsic value per share"
    )

    print(
        "Rows = Terminal Growth"
    )

    print(
        "Columns = WACC"
    )

    print(
        "=" * 80
    )

    print(
        analysis[
            "sensitivity"
        ].to_string()
    )


    # -----------------------------------------------------
    # MODEL NOTES
    # -----------------------------------------------------

    print(
        "\nMODEL NOTES:"
    )

    print(
        "• Historical accounting data is primarily sourced "
        "from SEC Company Facts."
    )

    print(
        "• Current market data and TTM statement data use "
        "Yahoo Finance."
    )

    print(
        "• Revenue growth uses both historical CAGR and "
        "recency-weighted historical growth."
    )

    print(
        "• Operating margins, D&A, CapEx, and working-capital "
        "investment are normalized from company history."
    )

    print(
        "• WACC uses beta, market capitalization, debt, taxes, "
        "and explicit market assumptions."
    )

    print(
        "• Reverse DCF estimates the growth required to justify "
        "the current market price under base assumptions."
    )

    print(
        "• Bear/Base/Bull values are scenario estimates, "
        "not guaranteed outcomes."
    )

    print(
        "• The DCF should be used as one part of a broader "
        "investment research process."
    )


if __name__ == "__main__":
    main()