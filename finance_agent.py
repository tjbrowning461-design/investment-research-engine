import json
import os
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


# =========================================================
# SETTINGS
# =========================================================

load_dotenv()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

MODEL = "gpt-5.6"


# =========================================================
# STRUCTURED AI OUTPUT
# =========================================================

class EarningsSentiment(BaseModel):
    transcript_available: bool

    overall_sentiment: str = Field(
        description=(
            "Positive, Slightly Positive, Neutral, "
            "Slightly Negative, Negative, or Not Analyzed"
        )
    )

    management_tone: str
    confidence_level: str
    guidance_tone: str

    key_positive_themes: list[str]
    key_concerns: list[str]
    potential_evasiveness: list[str]
    key_management_themes: list[str]

    sentiment_summary: str


class InvestmentBrief(BaseModel):
    executive_summary: str

    financial_health: str
    growth_analysis: str
    multi_quarter_trend_analysis: str
    profitability_analysis: str
    cash_flow_analysis: str
    balance_sheet_analysis: str
    valuation_analysis: str
    sec_filing_analysis: str

    positive_signals: list[str]
    key_risks: list[str]
    items_to_watch: list[str]

    earnings_sentiment: EarningsSentiment

    overall_risk_level: str = Field(
        description="Low, Moderate, High, or Very High"
    )

    bottom_line: str


# =========================================================
# BASIC HELPERS
# =========================================================

def get_row(statement, possible_names):

    if statement is None or statement.empty:
        return None

    for name in possible_names:

        if name in statement.index:
            return statement.loc[name]

    return None


def get_value_by_position(
    statement,
    possible_names,
    position,
):

    row = get_row(
        statement,
        possible_names,
    )

    if row is None:
        return None

    if not isinstance(
        row,
        pd.Series,
    ):
        return None

    valid_values = [
        float(value)
        for value in row
        if pd.notna(value)
    ]

    if len(valid_values) <= position:
        return None

    return valid_values[position]


def latest(
    statement,
    possible_names,
):

    return get_value_by_position(
        statement,
        possible_names,
        0,
    )


def safe_divide(
    numerator,
    denominator,
):

    if numerator is None:
        return None

    if denominator is None:
        return None

    if denominator == 0:
        return None

    return numerator / denominator


def growth(
    current,
    previous,
):

    if current is None:
        return None

    if previous is None:
        return None

    if previous == 0:
        return None

    return (
        current - previous
    ) / abs(previous)


def percentage_change_points(
    current,
    previous,
):

    if current is None:
        return None

    if previous is None:
        return None

    return current - previous


# =========================================================
# HISTORICAL HELPERS
# =========================================================

def get_statement_value_for_date(
    statement,
    possible_names,
    date_column,
):

    if (
        statement is None
        or statement.empty
    ):
        return None

    row = get_row(
        statement,
        possible_names,
    )

    if row is None:
        return None

    if date_column not in row.index:
        return None

    value = row[
        date_column
    ]

    if pd.isna(value):
        return None

    try:
        return float(value)

    except Exception:
        return None


def normalize_date(
    date_value
):

    try:

        return pd.Timestamp(
            date_value
        ).strftime(
            "%Y-%m-%d"
        )

    except Exception:

        return str(
            date_value
        )


def build_quarterly_history(
    income,
    balance,
    cash_flow,
    max_quarters=8,
):

    all_dates = set()

    for statement in [
        income,
        balance,
        cash_flow,
    ]:

        if (
            statement is not None
            and not statement.empty
        ):

            for column in statement.columns:
                all_dates.add(column)

    if not all_dates:
        return []

    sorted_dates = sorted(
        all_dates,
        reverse=True,
    )

    history = []

    for date_column in sorted_dates:

        revenue = get_statement_value_for_date(
            income,
            [
                "Total Revenue",
                "Operating Revenue",
            ],
            date_column,
        )

        gross_profit = get_statement_value_for_date(
            income,
            [
                "Gross Profit",
            ],
            date_column,
        )

        operating_income = get_statement_value_for_date(
            income,
            [
                "Operating Income",
            ],
            date_column,
        )

        pretax_income = get_statement_value_for_date(
            income,
            [
                "Pretax Income",
                "Pre Tax Income",
                "Income Before Tax",
            ],
            date_column,
        )

        tax_provision = get_statement_value_for_date(
            income,
            [
                "Tax Provision",
                "Income Tax Expense",
            ],
            date_column,
        )

        interest_expense = get_statement_value_for_date(
            income,
            [
                "Interest Expense Non Operating",
                "Interest Expense",
            ],
            date_column,
        )

        net_income = get_statement_value_for_date(
            income,
            [
                "Net Income",
                "Net Income Common Stockholders",
            ],
            date_column,
        )

        diluted_eps = get_statement_value_for_date(
            income,
            [
                "Diluted EPS",
                "Diluted EPS Continuous Operations",
            ],
            date_column,
        )

        cash = get_statement_value_for_date(
            balance,
            [
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Cash Equivalents",
                "Cash Financial",
            ],
            date_column,
        )

        total_assets = get_statement_value_for_date(
            balance,
            [
                "Total Assets",
            ],
            date_column,
        )

        total_debt = get_statement_value_for_date(
            balance,
            [
                "Total Debt",
            ],
            date_column,
        )

        equity = get_statement_value_for_date(
            balance,
            [
                "Stockholders Equity",
                "Common Stock Equity",
                "Total Equity Gross Minority Interest",
            ],
            date_column,
        )

        inventory = get_statement_value_for_date(
            balance,
            [
                "Inventory",
            ],
            date_column,
        )

        operating_cash_flow = get_statement_value_for_date(
            cash_flow,
            [
                "Operating Cash Flow",
                "Total Cash From Operating Activities",
            ],
            date_column,
        )

        capex = get_statement_value_for_date(
            cash_flow,
            [
                "Capital Expenditure",
                "Capital Expenditures",
            ],
            date_column,
        )

        depreciation_amortization = get_statement_value_for_date(
            cash_flow,
            [
                "Depreciation And Amortization",
                "Depreciation Amortization Depletion",
                "Depreciation",
            ],
            date_column,
        )

        free_cash_flow = None

        if (
            operating_cash_flow is not None
            and capex is not None
        ):

            free_cash_flow = (
                operating_cash_flow
                + capex
            )

        gross_margin = safe_divide(
            gross_profit,
            revenue,
        )

        operating_margin = safe_divide(
            operating_income,
            revenue,
        )

        net_margin = safe_divide(
            net_income,
            revenue,
        )

        fcf_margin = safe_divide(
            free_cash_flow,
            revenue,
        )

        ebitda = None

        if (
            operating_income is not None
            and depreciation_amortization is not None
        ):

            ebitda = (
                operating_income
                + depreciation_amortization
            )

        quarter_data = {

            "date":
                normalize_date(
                    date_column
                ),

            "revenue":
                revenue,

            "gross_profit":
                gross_profit,

            "operating_income":
                operating_income,

            "pretax_income":
                pretax_income,

            "tax_provision":
                tax_provision,

            "interest_expense":
                interest_expense,

            "net_income":
                net_income,

            "diluted_eps":
                diluted_eps,

            "cash":
                cash,

            "total_assets":
                total_assets,

            "total_debt":
                total_debt,

            "shareholders_equity":
                equity,

            "inventory":
                inventory,

            "operating_cash_flow":
                operating_cash_flow,

            "capital_expenditures":
                capex,

            "depreciation_amortization":
                depreciation_amortization,

            "free_cash_flow":
                free_cash_flow,

            "ebitda":
                ebitda,

            "gross_margin":
                gross_margin,

            "operating_margin":
                operating_margin,

            "net_margin":
                net_margin,

            "free_cash_flow_margin":
                fcf_margin,
        }

        core_values = [
            revenue,
            gross_profit,
            operating_income,
            net_income,
            free_cash_flow,
        ]

        if any(
            value is not None
            for value in core_values
        ):

            history.append(
                quarter_data
            )

        if len(history) >= max_quarters:
            break

    history = sorted(
        history,
        key=lambda item: item[
            "date"
        ],
    )

    return history


# =========================================================
# TTM / ADVANCED METRICS
# =========================================================

def sum_last_four(
    history,
    key,
):

    if len(history) < 4:
        return None

    quarters = history[-4:]

    values = [
        quarter.get(key)
        for quarter in quarters
    ]

    if any(
        value is None
        for value in values
    ):
        return None

    return sum(values)


def average_values(
    first_value,
    second_value,
):

    if first_value is None:
        return None

    if second_value is None:
        return None

    return (
        first_value
        + second_value
    ) / 2


def build_advanced_metrics(
    history,
    latest_balance,
    eps_yoy,
):

    ttm_revenue = sum_last_four(
        history,
        "revenue",
    )

    ttm_operating_income = sum_last_four(
        history,
        "operating_income",
    )

    ttm_net_income = sum_last_four(
        history,
        "net_income",
    )

    ttm_free_cash_flow = sum_last_four(
        history,
        "free_cash_flow",
    )

    ttm_ebitda = sum_last_four(
        history,
        "ebitda",
    )

    ttm_interest_expense = sum_last_four(
        history,
        "interest_expense",
    )

    ttm_pretax_income = sum_last_four(
        history,
        "pretax_income",
    )

    ttm_tax_provision = sum_last_four(
        history,
        "tax_provision",
    )

    ttm_diluted_eps = sum_last_four(
        history,
        "diluted_eps",
    )

    cash = latest_balance.get(
        "cash"
    )

    total_debt = latest_balance.get(
        "total_debt"
    )

    equity = latest_balance.get(
        "shareholders_equity"
    )

    total_assets = latest_balance.get(
        "total_assets"
    )

    net_debt = None

    if (
        total_debt is not None
        and cash is not None
    ):

        net_debt = (
            total_debt
            - cash
        )

    prior_reference = None

    if len(history) >= 5:
        prior_reference = history[-5]

    elif history:
        prior_reference = history[0]

    prior_assets = (
        prior_reference.get(
            "total_assets"
        )
        if prior_reference
        else None
    )

    prior_equity = (
        prior_reference.get(
            "shareholders_equity"
        )
        if prior_reference
        else None
    )

    average_assets = average_values(
        total_assets,
        prior_assets,
    )

    average_equity = average_values(
        equity,
        prior_equity,
    )

    roa = safe_divide(
        ttm_net_income,
        average_assets,
    )

    roe = safe_divide(
        ttm_net_income,
        average_equity,
    )

    effective_tax_rate = safe_divide(
        ttm_tax_provision,
        ttm_pretax_income,
    )

    if (
        effective_tax_rate is not None
        and (
            effective_tax_rate < 0
            or effective_tax_rate > 1
        )
    ):

        effective_tax_rate = None

    nopat = None

    if (
        ttm_operating_income is not None
        and effective_tax_rate is not None
    ):

        nopat = (
            ttm_operating_income
            * (
                1
                - effective_tax_rate
            )
        )

    invested_capital = None

    if (
        total_debt is not None
        and equity is not None
        and cash is not None
    ):

        invested_capital = (
            total_debt
            + equity
            - cash
        )

    prior_invested_capital = None

    if prior_reference:

        prior_debt = prior_reference.get(
            "total_debt"
        )

        prior_equity_value = prior_reference.get(
            "shareholders_equity"
        )

        prior_cash = prior_reference.get(
            "cash"
        )

        if (
            prior_debt is not None
            and prior_equity_value is not None
            and prior_cash is not None
        ):

            prior_invested_capital = (
                prior_debt
                + prior_equity_value
                - prior_cash
            )

    average_invested_capital = average_values(
        invested_capital,
        prior_invested_capital,
    )

    roic = safe_divide(
        nopat,
        average_invested_capital,
    )

    interest_coverage = None

    if (
        ttm_operating_income is not None
        and ttm_interest_expense is not None
        and ttm_interest_expense != 0
    ):

        interest_coverage = (
            ttm_operating_income
            / abs(
                ttm_interest_expense
            )
        )

    return {

        "ttm_revenue":
            ttm_revenue,

        "ttm_operating_income":
            ttm_operating_income,

        "ttm_net_income":
            ttm_net_income,

        "ttm_free_cash_flow":
            ttm_free_cash_flow,

        "ttm_ebitda":
            ttm_ebitda,

        "ttm_diluted_eps":
            ttm_diluted_eps,

        "net_debt":
            net_debt,

        "roa":
            roa,

        "roe":
            roe,

        "roic":
            roic,

        "interest_coverage":
            interest_coverage,

        "effective_tax_rate":
            effective_tax_rate,

        "diluted_eps_growth_yoy":
            eps_yoy,
    }


# =========================================================
# FORMATTERS
# =========================================================

def format_money(value):

    if value is None:
        return "Not available"

    absolute = abs(value)

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

    return f"${value:,.0f}"


def format_percent(value):

    if value is None:
        return "Not available"

    return f"{value * 100:,.2f}%"


def format_ratio(value):

    if value is None:
        return "Not available"

    return f"{value:,.2f}"


# =========================================================
# RULE-BASED SIGNALS
# =========================================================

def generate_flags(
    data
):

    positives = []
    warnings = []
    watch_items = []

    ratios = data[
        "ratios"
    ]

    trends = data[
        "growth"
    ]

    balance = data[
        "balance_sheet"
    ]

    revenue_yoy = trends.get(
        "revenue_growth_yoy"
    )

    net_income_yoy = trends.get(
        "net_income_growth_yoy"
    )

    fcf_yoy = trends.get(
        "free_cash_flow_growth_yoy"
    )

    gross_margin = ratios.get(
        "gross_margin"
    )

    operating_margin = ratios.get(
        "operating_margin"
    )

    current_ratio = ratios.get(
        "current_ratio"
    )

    debt_to_equity = ratios.get(
        "debt_to_equity"
    )

    fcf_margin = ratios.get(
        "free_cash_flow_margin"
    )

    if revenue_yoy is not None:

        if revenue_yoy >= 0.10:

            positives.append(
                "Revenue is growing more than 10% year over year."
            )

        elif revenue_yoy < 0:

            warnings.append(
                "Revenue declined year over year."
            )

        else:

            watch_items.append(
                "Revenue growth is positive but below 10% year over year."
            )

    if net_income_yoy is not None:

        if (
            revenue_yoy is not None
            and net_income_yoy > revenue_yoy
            and net_income_yoy > 0
        ):

            positives.append(
                "Net income is growing faster than revenue."
            )

        elif net_income_yoy < 0:

            warnings.append(
                "Net income declined year over year."
            )

    if fcf_yoy is not None:

        if fcf_yoy >= 0.10:

            positives.append(
                "Free cash flow increased more than 10% year over year."
            )

        elif fcf_yoy < 0:

            warnings.append(
                "Free cash flow declined year over year."
            )

    if gross_margin is not None:

        if gross_margin >= 0.40:

            positives.append(
                "Gross margin is above 40%."
            )

        elif gross_margin < 0.20:

            watch_items.append(
                "Gross margin is below 20%; industry context is important."
            )

    if operating_margin is not None:

        if operating_margin >= 0.20:

            positives.append(
                "Operating margin is above 20%."
            )

        elif operating_margin < 0:

            warnings.append(
                "The company is operating at a loss."
            )

    if fcf_margin is not None:

        if fcf_margin >= 0.15:

            positives.append(
                "Free cash flow margin is above 15%."
            )

        elif fcf_margin < 0:

            warnings.append(
                "Free cash flow is negative."
            )

    if current_ratio is not None:

        if current_ratio < 1:

            warnings.append(
                "Current liabilities exceed current assets."
            )

        elif current_ratio < 1.2:

            watch_items.append(
                "Current ratio is relatively close to 1.0."
            )

        elif current_ratio >= 1.5:

            positives.append(
                "The company has a relatively strong current ratio."
            )

    if debt_to_equity is not None:

        if debt_to_equity > 2:

            warnings.append(
                "Debt-to-equity is above 2.0."
            )

        elif debt_to_equity > 1:

            watch_items.append(
                "Debt exceeds shareholders' equity."
            )

    cash = balance.get(
        "cash"
    )

    debt = balance.get(
        "total_debt"
    )

    if (
        cash is not None
        and debt is not None
    ):

        if cash > debt:

            positives.append(
                "Cash exceeds total debt."
            )

        else:

            watch_items.append(
                "Total debt exceeds reported cash and short-term investments."
            )

    return {

        "positive_signals":
            positives,

        "warning_signals":
            warnings,

        "watch_items":
            watch_items,
    }


# =========================================================
# FINANCIAL DATA COLLECTION
# =========================================================

def get_financial_data(
    ticker_symbol
):

    ticker_symbol = (
        ticker_symbol
        .strip()
        .upper()
    )

    print(
        f"\nDownloading financial data for "
        f"{ticker_symbol}..."
    )

    company = yf.Ticker(
        ticker_symbol
    )

    try:
        info = company.info

    except Exception:
        info = {}

    company_name = (
        info.get(
            "longName"
        )
        or ticker_symbol
    )

    sector = info.get(
        "sector"
    )

    industry = info.get(
        "industry"
    )

    market_cap = info.get(
        "marketCap"
    )

    enterprise_value = info.get(
        "enterpriseValue"
    )

    trailing_pe = info.get(
        "trailingPE"
    )

    forward_pe = info.get(
        "forwardPE"
    )

    price_to_sales = info.get(
        "priceToSalesTrailing12Months"
    )

    price_to_book = info.get(
        "priceToBook"
    )

    enterprise_to_ebitda = info.get(
        "enterpriseToEbitda"
    )

    enterprise_to_revenue = info.get(
        "enterpriseToRevenue"
    )

    income = company.quarterly_income_stmt
    balance = company.quarterly_balance_sheet
    cash_flow = company.quarterly_cashflow


    # =====================================================
    # INCOME
    # =====================================================

    revenue = latest(
        income,
        [
            "Total Revenue",
            "Operating Revenue",
        ],
    )

    gross_profit = latest(
        income,
        [
            "Gross Profit",
        ],
    )

    operating_income = latest(
        income,
        [
            "Operating Income",
        ],
    )

    net_income = latest(
        income,
        [
            "Net Income",
            "Net Income Common Stockholders",
        ],
    )

    diluted_eps = latest(
        income,
        [
            "Diluted EPS",
            "Diluted EPS Continuous Operations",
        ],
    )

    prior_revenue = get_value_by_position(
        income,
        [
            "Total Revenue",
            "Operating Revenue",
        ],
        1,
    )

    prior_net_income = get_value_by_position(
        income,
        [
            "Net Income",
            "Net Income Common Stockholders",
        ],
        1,
    )

    revenue_last_year = get_value_by_position(
        income,
        [
            "Total Revenue",
            "Operating Revenue",
        ],
        4,
    )

    gross_profit_last_year = get_value_by_position(
        income,
        [
            "Gross Profit",
        ],
        4,
    )

    operating_income_last_year = get_value_by_position(
        income,
        [
            "Operating Income",
        ],
        4,
    )

    net_income_last_year = get_value_by_position(
        income,
        [
            "Net Income",
            "Net Income Common Stockholders",
        ],
        4,
    )

    diluted_eps_last_year = get_value_by_position(
        income,
        [
            "Diluted EPS",
            "Diluted EPS Continuous Operations",
        ],
        4,
    )


    # =====================================================
    # BALANCE
    # =====================================================

    cash = latest(
        balance,
        [
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents",
            "Cash Financial",
        ],
    )

    total_assets = latest(
        balance,
        [
            "Total Assets",
        ],
    )

    total_debt = latest(
        balance,
        [
            "Total Debt",
        ],
    )

    equity = latest(
        balance,
        [
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest",
        ],
    )

    current_assets = latest(
        balance,
        [
            "Current Assets",
            "Total Current Assets",
        ],
    )

    current_liabilities = latest(
        balance,
        [
            "Current Liabilities",
            "Total Current Liabilities",
        ],
    )

    inventory = latest(
        balance,
        [
            "Inventory",
        ],
    )

    inventory_last_year = get_value_by_position(
        balance,
        [
            "Inventory",
        ],
        4,
    )


    # =====================================================
    # CASH FLOW
    # =====================================================

    operating_cash_flow = latest(
        cash_flow,
        [
            "Operating Cash Flow",
            "Total Cash From Operating Activities",
        ],
    )

    capex = latest(
        cash_flow,
        [
            "Capital Expenditure",
            "Capital Expenditures",
        ],
    )

    free_cash_flow = None

    if (
        operating_cash_flow is not None
        and capex is not None
    ):

        free_cash_flow = (
            operating_cash_flow
            + capex
        )

    operating_cash_flow_last_year = get_value_by_position(
        cash_flow,
        [
            "Operating Cash Flow",
            "Total Cash From Operating Activities",
        ],
        4,
    )

    capex_last_year = get_value_by_position(
        cash_flow,
        [
            "Capital Expenditure",
            "Capital Expenditures",
        ],
        4,
    )

    free_cash_flow_last_year = None

    if (
        operating_cash_flow_last_year is not None
        and capex_last_year is not None
    ):

        free_cash_flow_last_year = (
            operating_cash_flow_last_year
            + capex_last_year
        )


    # =====================================================
    # RATIOS
    # =====================================================

    gross_margin = safe_divide(
        gross_profit,
        revenue,
    )

    operating_margin = safe_divide(
        operating_income,
        revenue,
    )

    net_margin = safe_divide(
        net_income,
        revenue,
    )

    fcf_margin = safe_divide(
        free_cash_flow,
        revenue,
    )

    debt_to_equity = safe_divide(
        total_debt,
        equity,
    )

    current_ratio = safe_divide(
        current_assets,
        current_liabilities,
    )

    gross_margin_last_year = safe_divide(
        gross_profit_last_year,
        revenue_last_year,
    )

    operating_margin_last_year = safe_divide(
        operating_income_last_year,
        revenue_last_year,
    )

    net_margin_last_year = safe_divide(
        net_income_last_year,
        revenue_last_year,
    )


    # =====================================================
    # GROWTH
    # =====================================================

    eps_yoy = growth(
        diluted_eps,
        diluted_eps_last_year,
    )


    # =====================================================
    # HISTORY
    # =====================================================

    quarterly_history = build_quarterly_history(
        income,
        balance,
        cash_flow,
        max_quarters=8,
    )


    # =====================================================
    # MAIN RESULT
    # =====================================================

    result = {

        "ticker":
            ticker_symbol,

        "company_name":
            company_name,

        "company": {

            "sector":
                sector,

            "industry":
                industry,

            "market_cap":
                market_cap,

            "enterprise_value":
                enterprise_value,
        },

        "income_statement": {

            "revenue":
                revenue,

            "gross_profit":
                gross_profit,

            "operating_income":
                operating_income,

            "net_income":
                net_income,

            "diluted_eps":
                diluted_eps,
        },

        "balance_sheet": {

            "cash":
                cash,

            "total_assets":
                total_assets,

            "total_debt":
                total_debt,

            "shareholders_equity":
                equity,

            "current_assets":
                current_assets,

            "current_liabilities":
                current_liabilities,

            "inventory":
                inventory,
        },

        "cash_flow": {

            "operating_cash_flow":
                operating_cash_flow,

            "capital_expenditures":
                capex,

            "free_cash_flow":
                free_cash_flow,
        },

        "ratios": {

            "gross_margin":
                gross_margin,

            "operating_margin":
                operating_margin,

            "net_profit_margin":
                net_margin,

            "free_cash_flow_margin":
                fcf_margin,

            "debt_to_equity":
                debt_to_equity,

            "current_ratio":
                current_ratio,
        },

        "growth": {

            "revenue_growth_qoq":
                growth(
                    revenue,
                    prior_revenue,
                ),

            "net_income_growth_qoq":
                growth(
                    net_income,
                    prior_net_income,
                ),

            "revenue_growth_yoy":
                growth(
                    revenue,
                    revenue_last_year,
                ),

            "gross_profit_growth_yoy":
                growth(
                    gross_profit,
                    gross_profit_last_year,
                ),

            "operating_income_growth_yoy":
                growth(
                    operating_income,
                    operating_income_last_year,
                ),

            "net_income_growth_yoy":
                growth(
                    net_income,
                    net_income_last_year,
                ),

            "diluted_eps_growth_yoy":
                eps_yoy,

            "free_cash_flow_growth_yoy":
                growth(
                    free_cash_flow,
                    free_cash_flow_last_year,
                ),

            "inventory_growth_yoy":
                growth(
                    inventory,
                    inventory_last_year,
                ),
        },

        "margin_changes": {

            "gross_margin_change_yoy":
                percentage_change_points(
                    gross_margin,
                    gross_margin_last_year,
                ),

            "operating_margin_change_yoy":
                percentage_change_points(
                    operating_margin,
                    operating_margin_last_year,
                ),

            "net_margin_change_yoy":
                percentage_change_points(
                    net_margin,
                    net_margin_last_year,
                ),
        },

        "valuation": {

            "market_cap":
                market_cap,

            "enterprise_value":
                enterprise_value,

            "trailing_pe":
                trailing_pe,

            "forward_pe":
                forward_pe,

            "price_to_sales":
                price_to_sales,

            "price_to_book":
                price_to_book,

            "enterprise_to_ebitda":
                enterprise_to_ebitda,

            "enterprise_to_revenue":
                enterprise_to_revenue,
        },

        "quarterly_history":
            quarterly_history,
    }

    result[
        "advanced_metrics"
    ] = build_advanced_metrics(
        quarterly_history,
        result[
            "balance_sheet"
        ],
        eps_yoy,
    )

    result[
        "signals"
    ] = generate_flags(
        result
    )

    return result


# =========================================================
# AI ANALYSIS
# =========================================================

def generate_ai_brief(
    data,
    earnings_transcript=None,
    sec_filing_context=None,
):

    if not os.getenv(
        "OPENAI_API_KEY"
    ):
        return None

    client = OpenAI()

    transcript_text = (
        earnings_transcript.strip()
        if earnings_transcript
        else ""
    )

    filing_text = (
        sec_filing_context.strip()
        if sec_filing_context
        else ""
    )

    payload = {

        "financial_data":
            data,

        "sec_filing_context":
            (
                filing_text
                if filing_text
                else "No SEC filing text supplied."
            ),

        "earnings_transcript":
            (
                transcript_text
                if transcript_text
                else "No earnings transcript supplied."
            ),
    }

    system_prompt = """
You are a financial analysis assistant.

Analyze ONLY the supplied:
1. structured financial data,
2. quarterly history,
3. SEC filing excerpts,
4. optional earnings-call transcript.

Never add factual claims from memory, outside sources, or assumptions.

=========================================================
SOURCE DISCIPLINE
=========================================================

Every factual claim must be supported by the supplied material.

If evidence is missing, state that it cannot be determined.

Analyst statements in a transcript are not automatically company facts.

Clearly distinguish:
- Python-calculated financial data
- management statements
- filing disclosures
- analyst questions
- your interpretation

=========================================================
FINANCIAL ANALYSIS
=========================================================

Evaluate:
- revenue
- earnings
- diluted EPS
- margins
- free cash flow
- liquidity
- leverage
- EBITDA when available
- ROA
- ROE
- ROIC
- net debt
- interest coverage
- valuation

Treat Python-calculated financial values as the numerical source of truth.

Do not invent missing metrics.

ROA, ROE, ROIC, EBITDA, and interest coverage may legitimately be
unavailable if the required source data is missing.

=========================================================
MULTI-QUARTER ANALYSIS
=========================================================

Use quarterly_history to assess direction across the supplied quarters.

Discuss:
- revenue direction
- profitability direction
- margin direction
- EPS direction
- free-cash-flow direction
- inventory direction
- debt direction

Account for seasonality.

Do NOT call a seasonal spike or decline a persistent trend without
enough evidence.

Do not imply eight quarters are available if fewer were supplied.

=========================================================
SEC ANALYSIS
=========================================================

Use only the supplied SEC excerpts.

Identify supported:
- business trends
- liquidity issues
- capital allocation
- operating changes
- risks
- uncertainties

Do not claim that a risk is new or worsening unless supported.

=========================================================
EARNINGS TRANSCRIPT
=========================================================

If a substantive transcript is supplied, assess:
- sentiment
- tone
- confidence
- guidance
- positive themes
- concerns
- repeated themes
- potential evasiveness

Do not classify normal caution, refusing to speculate, or declining
long-range guidance as evasiveness.

If no substantive transcript is supplied:

transcript_available = false
overall_sentiment = "Not Analyzed"
management_tone = "Not analyzed — no substantive transcript supplied."
confidence_level = "Not analyzed — no substantive transcript supplied."
guidance_tone = "Not analyzed — no substantive transcript supplied."
key_positive_themes = []
key_concerns = []
potential_evasiveness = []
key_management_themes = []

=========================================================
VALUATION
=========================================================

Separate business quality from stock valuation.

Do not give:
- buy recommendations
- sell recommendations
- hold recommendations
- price targets
- stock-price predictions

High valuation multiples are not proof of overvaluation without an
appropriate benchmark.

=========================================================
UNCERTAINTY
=========================================================

Prefer saying:
"The supplied information does not establish..."
"There is insufficient evidence..."
"This metric is unavailable..."
"The filing excerpt does not support..."

rather than filling gaps with assumptions.
"""

    response = client.responses.parse(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    indent=2,
                ),
            },
        ],
        text_format=InvestmentBrief,
    )

    return response.output_parsed


# =========================================================
# SAVE
# =========================================================

def save_financial_data(
    data,
    ai_brief=None,
):

    ticker = data[
        "ticker"
    ]

    result = {

        "financial_data":
            data,

        "ai_brief":
            (
                ai_brief.model_dump()
                if ai_brief
                else None
            ),
    }

    output_file = (
        DATA_DIR
        / f"{ticker}_financial_brief.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    return output_file