import json
import math
import os
import sys
from pathlib import Path

from openai import OpenAI

from research_engine import (
    build_research_package,
    make_serializable,
)


# =========================================================
# PROJECT PATHS
# =========================================================

DESKTOP = Path.home() / "Desktop"

COMPARISON_ENGINE_DIR = (
    DESKTOP
    / "company-comparison-engine"
)

comparison_path = str(
    COMPARISON_ENGINE_DIR
)

if comparison_path not in sys.path:
    sys.path.insert(
        0,
        comparison_path,
    )


from comparison_engine import (
    compare_companies,
)


# =========================================================
# SETTINGS
# =========================================================

MODEL = "gpt-5.6"


# =========================================================
# DEFAULT RESEARCH COMPARISON GROUPS
# =========================================================
#
# These are financial comparison groups.
# They are NOT claims that every company listed is a
# direct product competitor.
# =========================================================

DEFAULT_PEERS = {
    "AAPL": [
        "MSFT",
        "GOOGL",
        "META",
        "AMZN",
        "NVDA",
    ],

    "MSFT": [
        "AAPL",
        "GOOGL",
        "META",
        "AMZN",
        "NVDA",
    ],

    "GOOGL": [
        "META",
        "MSFT",
        "AMZN",
        "AAPL",
        "NVDA",
    ],

    "META": [
        "GOOGL",
        "MSFT",
        "AMZN",
        "AAPL",
        "NVDA",
    ],

    "AMZN": [
        "MSFT",
        "GOOGL",
        "META",
        "AAPL",
        "NVDA",
    ],

    "NVDA": [
        "AMD",
        "MSFT",
        "GOOGL",
        "META",
        "AAPL",
    ],

    "AMD": [
        "NVDA",
        "MSFT",
        "GOOGL",
        "AAPL",
        "META",
    ],

    "TSLA": [
        "AAPL",
        "NVDA",
        "AMZN",
        "MSFT",
        "GOOGL",
    ],

    "NFLX": [
        "META",
        "GOOGL",
        "AMZN",
        "MSFT",
        "AAPL",
    ],

    "CRM": [
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "AAPL",
    ],
}


# =========================================================
# BASIC HELPERS
# =========================================================

def clamp(
    value,
    minimum=0.0,
    maximum=100.0,
):
    if value is None:
        return None

    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


def average_available(
    values,
):
    clean = [
        float(value)
        for value in values
        if value is not None
    ]

    if not clean:
        return None

    return sum(clean) / len(clean)


def weighted_average_available(
    weighted_values,
):
    """
    weighted_values:
        [
            (value, weight),
            (value, weight),
        ]

    Missing values are excluded and remaining
    weights are automatically re-normalized.
    """

    total = 0.0
    used_weight = 0.0

    for value, weight in weighted_values:

        if value is None:
            continue

        total += (
            float(value)
            * float(weight)
        )

        used_weight += float(
            weight
        )

    if used_weight == 0:
        return None

    return total / used_weight


def score_linear(
    value,
    bad,
    good,
):
    """
    Maps a metric to a 0-100 score.

    Works for both:
    higher-is-better:
        bad < good

    lower-is-better:
        bad > good
    """

    if value is None:
        return None

    value = float(value)

    if good == bad:
        return 50.0

    score = (
        (
            value
            - bad
        )
        / (
            good
            - bad
        )
        * 100
    )

    return clamp(score)


def confidence_multiplier(
    confidence,
):
    mapping = {
        "High":
            1.00,

        "Moderate":
            0.75,

        "Low":
            0.50,

        "Limited":
            0.40,
    }

    return mapping.get(
        confidence,
        0.60,
    )


def round_or_none(
    value,
    digits=1,
):
    if value is None:
        return None

    return round(
        float(value),
        digits,
    )


# =========================================================
# PEER PERCENTILE
# =========================================================

def percentile_score(
    target_value,
    peer_values,
    higher_is_better=True,
):
    """
    Relative 0-100 score.

    A score near 100 means the target ranks toward
    the favorable end of the supplied peer group.

    This is not an economic-moat score.
    """

    if target_value is None:
        return None

    clean = [
        float(value)
        for value in peer_values
        if value is not None
    ]

    if not clean:
        return None

    target_value = float(
        target_value
    )

    below = sum(
        value < target_value
        for value in clean
    )

    equal = sum(
        value == target_value
        for value in clean
    )

    percentile = (
        below
        + 0.5 * equal
    ) / len(clean)

    score = percentile * 100

    if not higher_is_better:
        score = 100 - score

    return clamp(score)


# =========================================================
# GROWTH
# =========================================================

def score_growth(
    comparison,
    dcf,
):
    revenue_growth = comparison.get(
        "revenue_growth_yoy_percent"
    )

    operating_growth = comparison.get(
        "operating_income_growth_yoy_percent"
    )

    net_income_growth = comparison.get(
        "net_income_growth_yoy_percent"
    )

    weighted_growth = dcf.get(
        "weighted_recent_growth_percent"
    )

    revenue_score = score_linear(
        revenue_growth,
        bad=-10,
        good=25,
    )

    operating_score = score_linear(
        operating_growth,
        bad=-15,
        good=30,
    )

    net_income_score = score_linear(
        net_income_growth,
        bad=-15,
        good=30,
    )

    historical_score = score_linear(
        weighted_growth,
        bad=0,
        good=20,
    )


    score = weighted_average_available(
        [
            (
                revenue_score,
                0.35,
            ),
            (
                operating_score,
                0.25,
            ),
            (
                net_income_score,
                0.15,
            ),
            (
                historical_score,
                0.25,
            ),
        ]
    )


    return {
        "score":
            score,

        "revenue_growth_score":
            revenue_score,

        "operating_growth_score":
            operating_score,

        "net_income_growth_score":
            net_income_score,

        "historical_growth_score":
            historical_score,

        "notes": [
            f"Revenue YoY growth: {revenue_growth}%",
            f"Operating income YoY growth: {operating_growth}%",
            f"Net income YoY growth: {net_income_growth}%",
            f"Weighted recent historical growth: {weighted_growth}%",
        ],
    }


# =========================================================
# PROFITABILITY
# =========================================================

def score_profitability(
    comparison,
):
    gross_margin = comparison.get(
        "gross_margin_percent"
    )

    operating_margin = comparison.get(
        "operating_margin_percent"
    )

    net_margin = comparison.get(
        "net_margin_percent"
    )


    gross_score = score_linear(
        gross_margin,
        bad=15,
        good=70,
    )

    operating_score = score_linear(
        operating_margin,
        bad=5,
        good=35,
    )

    net_score = score_linear(
        net_margin,
        bad=3,
        good=30,
    )


    score = weighted_average_available(
        [
            (
                gross_score,
                0.25,
            ),
            (
                operating_score,
                0.45,
            ),
            (
                net_score,
                0.30,
            ),
        ]
    )


    return {
        "score":
            score,

        "gross_margin_score":
            gross_score,

        "operating_margin_score":
            operating_score,

        "net_margin_score":
            net_score,

        "notes": [
            f"Gross margin: {gross_margin}%",
            f"Operating margin: {operating_margin}%",
            f"Net margin: {net_margin}%",
        ],
    }


# =========================================================
# CASH FLOW
# =========================================================

def score_cash_flow(
    comparison,
):
    fcf_margin = comparison.get(
        "fcf_margin_percent"
    )

    score = score_linear(
        fcf_margin,
        bad=0,
        good=30,
    )


    return {
        "score":
            score,

        "fcf_margin_score":
            score,

        "notes": [
            f"Free-cash-flow margin: {fcf_margin}%"
        ],
    }


# =========================================================
# BALANCE SHEET
# =========================================================

def score_balance_sheet(
    comparison,
):
    debt_to_equity = comparison.get(
        "debt_to_equity"
    )

    current_ratio = comparison.get(
        "current_ratio"
    )


    debt_score = score_linear(
        debt_to_equity,
        bad=2.0,
        good=0.0,
    )


    current_score = score_linear(
        current_ratio,
        bad=0.6,
        good=2.0,
    )


    score = weighted_average_available(
        [
            (
                debt_score,
                0.55,
            ),
            (
                current_score,
                0.45,
            ),
        ]
    )


    return {
        "score":
            score,

        "debt_score":
            debt_score,

        "liquidity_score":
            current_score,

        "notes": [
            f"Debt / Equity: {debt_to_equity}",
            f"Current Ratio: {current_ratio}",
        ],
    }


# =========================================================
# ACCOUNTING QUALITY
# =========================================================

def score_accounting_quality(
    comparison,
):
    """
    No flags does NOT automatically mean 100.

    We start neutral-positive and adjust for the
    quality of the actual evidence.
    """

    score = 70.0

    notes = []

    data_quality = comparison.get(
        "data_quality"
    )


    if data_quality == "High":

        score += 15

        notes.append(
            "Underlying financial-data quality is High."
        )

    elif data_quality == "Moderate":

        score += 5

        notes.append(
            "Underlying financial-data quality is Moderate."
        )

    elif data_quality == "Limited":

        score -= 15

        notes.append(
            "Underlying financial-data quality is Limited."
        )


    if comparison.get(
        "unusual_non_operating"
    ):

        score -= 25

        ratio = comparison.get(
            "non_operating_ratio_percent"
        )

        notes.append(
            (
                "Material non-operating contribution detected"
                + (
                    f": {ratio}%."
                    if ratio is not None
                    else "."
                )
            )
        )

    else:

        score += 5

        notes.append(
            "No large non-operating distortion detected."
        )


    if comparison.get(
        "eps_comparability_warning"
    ):

        score -= 15

        notes.append(
            "EPS comparability warning detected."
        )

    else:

        score += 5


    missing_metrics = comparison.get(
        "missing_metrics",
        [],
    )


    if missing_metrics:

        deduction = min(
            len(missing_metrics)
            * 5,
            20,
        )

        score -= deduction

        notes.append(
            (
                "Missing metrics: "
                + ", ".join(
                    missing_metrics
                )
            )
        )


    return {
        "score":
            clamp(score),

        "notes":
            notes,
    }


# =========================================================
# INTRINSIC BUSINESS QUALITY
# =========================================================

def calculate_business_quality(
    scores,
):
    """
    Business Quality deliberately excludes:

    - peer ranking
    - valuation multiples
    - DCF

    A great company can still be an expensive stock.
    """

    score = weighted_average_available(
        [
            (
                scores[
                    "growth"
                ][
                    "score"
                ],
                0.20,
            ),
            (
                scores[
                    "profitability"
                ][
                    "score"
                ],
                0.25,
            ),
            (
                scores[
                    "cash_flow"
                ][
                    "score"
                ],
                0.20,
            ),
            (
                scores[
                    "balance_sheet"
                ][
                    "score"
                ],
                0.15,
            ),
            (
                scores[
                    "accounting_quality"
                ][
                    "score"
                ],
                0.20,
            ),
        ]
    )


    return {
        "score":
            score,

        "notes": [
            (
                "Intrinsic business-quality composite. "
                "Peer ranking and valuation are excluded."
            )
        ],
    }


# =========================================================
# PEER ANALYSIS
# =========================================================

def build_peer_analysis(
    ticker,
    peers=None,
):
    ticker = (
        ticker
        .strip()
        .upper()
    )


    if peers is None:
        peers = DEFAULT_PEERS.get(
            ticker,
            [],
        )


    cleaned_peers = []

    for peer in peers:

        peer = (
            peer
            .strip()
            .upper()
        )

        if (
            peer
            and peer != ticker
            and peer not in cleaned_peers
        ):

            cleaned_peers.append(
                peer
            )


    tickers = [
        ticker
    ] + cleaned_peers


    results = compare_companies(
        tickers,
        max_companies=10,
    )


    target = None
    peer_results = []


    for company in results:

        company_ticker = (
            company
            .get(
                "ticker",
                "",
            )
            .upper()
        )

        if company_ticker == ticker:
            target = company

        else:
            peer_results.append(
                company
            )


    return {
        "ticker":
            ticker,

        "peer_tickers":
            [
                company.get(
                    "ticker"
                )
                for company
                in peer_results
            ],

        "target":
            target,

        "peers":
            peer_results,
    }


# =========================================================
# PEER STRENGTH
# =========================================================

def score_peer_strength(
    peer_analysis,
):
    target = peer_analysis.get(
        "target"
    )

    peers = peer_analysis.get(
        "peers",
        [],
    )


    if not target or not peers:

        return {
            "score":
                None,

            "metric_scores":
                {},

            "notes": [
                "Peer comparison unavailable."
            ],
        }


    metric_definitions = {
        "revenue_growth_yoy":
            True,

        "operating_income_growth_yoy":
            True,

        "gross_margin":
            True,

        "operating_margin":
            True,

        "net_margin":
            True,

        "fcf_margin":
            True,

        "current_ratio":
            True,

        "debt_to_equity":
            False,
    }


    metric_scores = {}


    for metric, higher_is_better in (
        metric_definitions.items()
    ):

        target_value = target.get(
            metric
        )

        peer_values = [
            peer.get(
                metric
            )
            for peer
            in peers
        ]


        metric_scores[
            metric
        ] = percentile_score(
            target_value,
            peer_values,
            higher_is_better=(
                higher_is_better
            ),
        )


    # Operating economics matter more than
    # current-ratio/leverage peer ranking.

    score = weighted_average_available(
        [
            (
                metric_scores.get(
                    "revenue_growth_yoy"
                ),
                0.18,
            ),
            (
                metric_scores.get(
                    "operating_income_growth_yoy"
                ),
                0.15,
            ),
            (
                metric_scores.get(
                    "gross_margin"
                ),
                0.12,
            ),
            (
                metric_scores.get(
                    "operating_margin"
                ),
                0.18,
            ),
            (
                metric_scores.get(
                    "net_margin"
                ),
                0.12,
            ),
            (
                metric_scores.get(
                    "fcf_margin"
                ),
                0.15,
            ),
            (
                metric_scores.get(
                    "current_ratio"
                ),
                0.05,
            ),
            (
                metric_scores.get(
                    "debt_to_equity"
                ),
                0.05,
            ),
        ]
    )


    return {
        "score":
            score,

        "metric_scores":
            metric_scores,

        "notes": [
            (
                "Peer Strength measures relative financial "
                "performance within the supplied comparison set."
            ),
            (
                "It is not an economic-moat or competitive-advantage score."
            ),
        ],
    }


# =========================================================
# ABSOLUTE VALUATION
# =========================================================

def score_absolute_valuation(
    comparison,
):
    forward_pe = comparison.get(
        "forward_pe"
    )

    price_to_sales = comparison.get(
        "price_to_sales"
    )

    ev_to_ebitda = comparison.get(
        "enterprise_to_ebitda"
    )


    pe_score = score_linear(
        forward_pe,
        bad=50,
        good=12,
    )

    sales_score = score_linear(
        price_to_sales,
        bad=15,
        good=2,
    )

    ebitda_score = score_linear(
        ev_to_ebitda,
        bad=40,
        good=10,
    )


    score = weighted_average_available(
        [
            (
                pe_score,
                0.40,
            ),
            (
                sales_score,
                0.25,
            ),
            (
                ebitda_score,
                0.35,
            ),
        ]
    )


    return {
        "score":
            score,

        "forward_pe_score":
            pe_score,

        "price_to_sales_score":
            sales_score,

        "ev_ebitda_score":
            ebitda_score,

        "notes": [
            f"Forward P/E: {forward_pe}",
            f"Price / Sales: {price_to_sales}",
            f"EV / EBITDA: {ev_to_ebitda}",
        ],
    }


# =========================================================
# PEER-RELATIVE VALUATION
# =========================================================

def score_peer_valuation(
    peer_analysis,
):
    target = peer_analysis.get(
        "target"
    )

    peers = peer_analysis.get(
        "peers",
        [],
    )


    if not target or not peers:

        return {
            "score":
                None,

            "metric_scores":
                {},

            "notes": [
                "Peer-relative valuation unavailable."
            ],
        }


    metrics = [
        "forward_pe",
        "price_to_sales",
        "enterprise_to_ebitda",
        "enterprise_to_revenue",
    ]


    metric_scores = {}


    for metric in metrics:

        target_value = target.get(
            metric
        )

        peer_values = [
            peer.get(
                metric
            )
            for peer
            in peers
        ]


        metric_scores[
            metric
        ] = percentile_score(
            target_value,
            peer_values,
            higher_is_better=False,
        )


    score = weighted_average_available(
        [
            (
                metric_scores.get(
                    "forward_pe"
                ),
                0.35,
            ),
            (
                metric_scores.get(
                    "price_to_sales"
                ),
                0.20,
            ),
            (
                metric_scores.get(
                    "enterprise_to_ebitda"
                ),
                0.30,
            ),
            (
                metric_scores.get(
                    "enterprise_to_revenue"
                ),
                0.15,
            ),
        ]
    )


    return {
        "score":
            score,

        "metric_scores":
            metric_scores,

        "notes": [
            (
                "Higher score means cheaper valuation relative "
                "to the supplied comparison group."
            ),
            (
                "Peer-relative cheapness is not the same as "
                "intrinsic undervaluation."
            ),
        ],
    }


# =========================================================
# MULTIPLE-BASED VALUATION
# =========================================================

def score_multiple_valuation(
    comparison,
    peer_analysis,
):
    absolute = score_absolute_valuation(
        comparison
    )

    peer_relative = score_peer_valuation(
        peer_analysis
    )


    absolute_score = absolute.get(
        "score"
    )

    peer_score = peer_relative.get(
        "score"
    )


    score = weighted_average_available(
        [
            (
                absolute_score,
                0.45,
            ),
            (
                peer_score,
                0.55,
            ),
        ]
    )


    return {
        "score":
            score,

        "absolute_score":
            absolute_score,

        "peer_relative_score":
            peer_score,

        "notes":
            (
                absolute.get(
                    "notes",
                    [],
                )
                + peer_relative.get(
                    "notes",
                    [],
                )
            ),
    }


# =========================================================
# DCF VALUATION
# =========================================================

def score_dcf(
    dcf,
):
    base_upside = dcf.get(
        "base_upside_downside_percent"
    )

    current_price = dcf.get(
        "current_price"
    )

    bear_value = dcf.get(
        "bear_value"
    )

    bull_value = dcf.get(
        "bull_value"
    )


    base_score = score_linear(
        base_upside,
        bad=-50,
        good=50,
    )


    bull_upside = None

    if (
        current_price not in (
            None,
            0,
        )
        and bull_value is not None
    ):

        bull_upside = (
            (
                bull_value
                / current_price
            )
            - 1
        ) * 100


    bear_upside = None

    if (
        current_price not in (
            None,
            0,
        )
        and bear_value is not None
    ):

        bear_upside = (
            (
                bear_value
                / current_price
            )
            - 1
        ) * 100


    bull_score = score_linear(
        bull_upside,
        bad=-40,
        good=40,
    )

    bear_score = score_linear(
        bear_upside,
        bad=-80,
        good=0,
    )


    raw_score = weighted_average_available(
        [
            (
                base_score,
                0.50,
            ),
            (
                bull_score,
                0.30,
            ),
            (
                bear_score,
                0.20,
            ),
        ]
    )


    dcf_confidence = dcf.get(
        "overall_dcf_confidence"
    )

    multiplier = confidence_multiplier(
        dcf_confidence
    )


    # -----------------------------------------------------
    # Confidence-adjust toward neutral.
    #
    # A low-confidence DCF is still evidence,
    # but it cannot act like certainty.
    # -----------------------------------------------------

    if raw_score is None:

        adjusted_score = None

    else:

        adjusted_score = (
            50
            + (
                raw_score
                - 50
            )
            * multiplier
        )


    return {
        "score":
            clamp(
                adjusted_score
            )
            if adjusted_score is not None
            else None,

        "raw_score":
            raw_score,

        "confidence_multiplier":
            multiplier,

        "base_upside_percent":
            base_upside,

        "bull_upside_percent":
            bull_upside,

        "bear_upside_percent":
            bear_upside,

        "notes": [
            (
                f"Base DCF upside/downside: "
                f"{round_or_none(base_upside, 2)}%"
            ),
            (
                f"Bull DCF upside/downside: "
                f"{round_or_none(bull_upside, 2)}%"
            ),
            (
                f"Bear DCF upside/downside: "
                f"{round_or_none(bear_upside, 2)}%"
            ),
            (
                f"DCF confidence: "
                f"{dcf_confidence}"
            ),
        ],
    }


# =========================================================
# FINAL VALUATION SUPPORT
# =========================================================

def calculate_valuation_support(
    scores,
):
    """
    This is the key V3 correction.

    Multiple valuation and DCF are components
    of ONE investment-valuation judgment.

    They no longer get two independent votes
    capable of automatically forcing SELL.
    """

    multiple_score = (
        scores[
            "multiple_valuation"
        ][
            "score"
        ]
    )

    dcf_score = (
        scores[
            "dcf"
        ][
            "score"
        ]
    )


    score = weighted_average_available(
        [
            (
                multiple_score,
                0.45,
            ),
            (
                dcf_score,
                0.55,
            ),
        ]
    )


    return {
        "score":
            score,

        "multiple_valuation_score":
            multiple_score,

        "dcf_score":
            dcf_score,

        "notes": [
            (
                "Combines market-multiple valuation and "
                "confidence-adjusted DCF into one valuation-support score."
            ),
            (
                "This prevents valuation and DCF from double-counting "
                "the same expensive/cheap stock signal."
            ),
        ],
    }


# =========================================================
# MODEL / EVIDENCE CONFIDENCE
# =========================================================

def score_model_confidence(
    dcf,
    comparison,
    peer_analysis,
):
    score = 100.0

    notes = []


    forecast_uncertainty = dcf.get(
        "forecast_uncertainty"
    )

    terminal_reliance = dcf.get(
        "terminal_value_reliance"
    )

    dcf_confidence = dcf.get(
        "overall_dcf_confidence"
    )


    if forecast_uncertainty == "Moderate":

        score -= 15

    elif forecast_uncertainty == "High":

        score -= 30


    if terminal_reliance == "Moderate":

        score -= 10

    elif terminal_reliance == "High":

        score -= 25


    if dcf_confidence == "Moderate":

        score -= 10

    elif dcf_confidence == "Low":

        score -= 25


    data_quality = comparison.get(
        "data_quality"
    )


    if data_quality == "Moderate":

        score -= 5

    elif data_quality == "Limited":

        score -= 20


    peer_count = len(
        peer_analysis.get(
            "peers",
            [],
        )
    )


    if peer_count == 0:

        score -= 10

        notes.append(
            "No peer comparison available."
        )

    elif peer_count < 3:

        score -= 5

        notes.append(
            "Peer sample is small."
        )


    notes.extend(
        [
            (
                f"Forecast uncertainty: "
                f"{forecast_uncertainty}"
            ),
            (
                f"Terminal-value reliance: "
                f"{terminal_reliance}"
            ),
            (
                f"DCF confidence: "
                f"{dcf_confidence}"
            ),
            (
                f"Fundamental data quality: "
                f"{data_quality}"
            ),
            (
                f"Available peer count: "
                f"{peer_count}"
            ),
        ]
    )


    return {
        "score":
            clamp(score),

        "notes":
            notes,
    }


# =========================================================
# COMPLETE SCORE PACKAGE
# =========================================================

def calculate_scores(
    research_package,
    peer_analysis,
):
    comparison = (
        research_package.get(
            "comparison"
        )
        or {}
    )

    dcf = (
        research_package.get(
            "dcf"
        )
        or {}
    )


    scores = {}


    scores[
        "growth"
    ] = score_growth(
        comparison,
        dcf,
    )


    scores[
        "profitability"
    ] = score_profitability(
        comparison
    )


    scores[
        "cash_flow"
    ] = score_cash_flow(
        comparison
    )


    scores[
        "balance_sheet"
    ] = score_balance_sheet(
        comparison
    )


    scores[
        "accounting_quality"
    ] = score_accounting_quality(
        comparison
    )


    scores[
        "business_quality"
    ] = calculate_business_quality(
        scores
    )


    scores[
        "peer_strength"
    ] = score_peer_strength(
        peer_analysis
    )


    scores[
        "multiple_valuation"
    ] = score_multiple_valuation(
        comparison,
        peer_analysis,
    )


    scores[
        "dcf"
    ] = score_dcf(
        dcf
    )


    scores[
        "valuation_support"
    ] = calculate_valuation_support(
        scores
    )


    scores[
        "model_confidence"
    ] = score_model_confidence(
        dcf,
        comparison,
        peer_analysis,
    )


    return scores


# =========================================================
# FINAL OVERALL SCORE
# =========================================================
#
# Business quality is the largest pillar.
#
# Valuation Support contains both:
# - market-multiple valuation
# - DCF
#
# Therefore DCF is NOT separately weighted here.
# =========================================================

OVERALL_WEIGHTS = {
    "business_quality":
        0.50,

    "peer_strength":
        0.10,

    "valuation_support":
        0.30,

    "model_confidence":
        0.10,
}


def calculate_overall_score(
    scores,
):
    weighted_items = []


    for category, weight in (
        OVERALL_WEIGHTS.items()
    ):

        score = (
            scores.get(
                category,
                {}
            )
            .get(
                "score"
            )
        )

        weighted_items.append(
            (
                score,
                weight,
            )
        )


    return weighted_average_available(
        weighted_items
    )


# =========================================================
# RATING LOGIC
# =========================================================

def determine_rating(
    overall_score,
    scores,
):
    """
    Final V3 rating architecture.

    Key principle:

    Strong business + expensive stock
    is normally HOLD, not automatically SELL.

    SELL requires broader deterioration or a
    genuinely poor total risk/reward profile.
    """

    if overall_score is None:

        return {
            "rating":
                "HOLD",

            "reason":
                (
                    "Insufficient evidence for a stronger rating."
                ),
        }


    business_quality = (
        scores[
            "business_quality"
        ][
            "score"
        ]
    )

    peer_strength = (
        scores[
            "peer_strength"
        ][
            "score"
        ]
    )

    valuation_support = (
        scores[
            "valuation_support"
        ][
            "score"
        ]
    )

    accounting_quality = (
        scores[
            "accounting_quality"
        ][
            "score"
        ]
    )

    growth = (
        scores[
            "growth"
        ][
            "score"
        ]
    )

    profitability = (
        scores[
            "profitability"
        ][
            "score"
        ]
    )

    cash_flow = (
        scores[
            "cash_flow"
        ][
            "score"
        ]
    )


    # =====================================================
    # BUY
    # =====================================================
    #
    # Requires:
    # strong company
    # + valuation support
    # + acceptable accounting quality
    # + strong overall evidence.
    # =====================================================

    if (
        overall_score >= 68
        and business_quality is not None
        and business_quality >= 67
        and valuation_support is not None
        and valuation_support >= 55
        and accounting_quality is not None
        and accounting_quality >= 60
    ):

        return {
            "rating":
                "BUY",

            "reason":
                (
                    "Strong intrinsic business quality is supported "
                    "by an attractive enough valuation/risk-reward profile."
                ),
        }


    # =====================================================
    # SELL — VERY WEAK TOTAL PROFILE
    # =====================================================

    if overall_score <= 38:

        return {
            "rating":
                "SELL",

            "reason":
                (
                    "The overall research score indicates broadly "
                    "unfavorable business quality and/or investment risk-reward."
                ),
        }


    # =====================================================
    # SELL — WEAK BUSINESS + WEAK VALUATION
    # =====================================================

    if (
        business_quality is not None
        and business_quality <= 42
        and valuation_support is not None
        and valuation_support <= 40
    ):

        return {
            "rating":
                "SELL",

            "reason":
                (
                    "Weak business quality is combined with insufficient "
                    "valuation support."
                ),
        }


    # =====================================================
    # SELL — OPERATING DETERIORATION
    # =====================================================

    weak_operating_signals = sum(
        [
            (
                growth is not None
                and growth <= 35
            ),
            (
                profitability is not None
                and profitability <= 35
            ),
            (
                cash_flow is not None
                and cash_flow <= 35
            ),
        ]
    )


    if (
        business_quality is not None
        and business_quality <= 50
        and weak_operating_signals >= 2
    ):

        return {
            "rating":
                "SELL",

            "reason":
                (
                    "Multiple core operating indicators are weak "
                    "alongside below-average business quality."
                ),
        }


    # =====================================================
    # HOLD — STRONG BUSINESS / EXPENSIVE STOCK
    # =====================================================

    if (
        business_quality is not None
        and business_quality >= 65
        and valuation_support is not None
        and valuation_support < 45
    ):

        return {
            "rating":
                "HOLD",

            "reason":
                (
                    "The company scores strongly on intrinsic business quality, "
                    "but current valuation offers insufficient support for a Buy."
                ),
        }


    # =====================================================
    # HOLD — DEFAULT MIXED EVIDENCE
    # =====================================================

    return {
        "rating":
            "HOLD",

        "reason":
            (
                "The evidence is mixed: neither the positive nor negative "
                "case is strong enough to satisfy the Buy or Sell gates."
            ),
    }


# =========================================================
# RATING CONFIDENCE
# =========================================================

def calculate_rating_confidence(
    scores,
    overall_score,
):
    """
    Evidence confidence.

    NOT:
    probability the rating is correct.
    """

    core_categories = [
        "business_quality",
        "peer_strength",
        "valuation_support",
        "model_confidence",
    ]


    available_scores = []


    for category in core_categories:

        score = (
            scores.get(
                category,
                {}
            )
            .get(
                "score"
            )
        )

        if score is not None:

            available_scores.append(
                score
            )


    if not available_scores:

        return 25.0


    # -----------------------------------------------------
    # DATA COVERAGE
    # -----------------------------------------------------

    coverage = (
        len(
            available_scores
        )
        / len(
            core_categories
        )
    )


    # -----------------------------------------------------
    # AGREEMENT
    # -----------------------------------------------------

    mean_score = (
        sum(
            available_scores
        )
        / len(
            available_scores
        )
    )


    variance = (
        sum(
            (
                score
                - mean_score
            )
            ** 2
            for score
            in available_scores
        )
        / len(
            available_scores
        )
    )


    standard_deviation = math.sqrt(
        variance
    )


    agreement = clamp(
        100
        - standard_deviation
        * 2
    )


    # -----------------------------------------------------
    # MODEL QUALITY
    # -----------------------------------------------------

    evidence_quality = (
        scores[
            "model_confidence"
        ][
            "score"
        ]
    )


    if evidence_quality is None:

        evidence_quality = 50


    # -----------------------------------------------------
    # DECISIVENESS
    # -----------------------------------------------------
    #
    # Close to neutral = less confidence in a categorical call.
    # -----------------------------------------------------

    decisiveness = min(
        abs(
            overall_score
            - 50
        )
        * 2.5,
        100,
    )


    confidence = (
        coverage
        * 25
        + agreement
        * 0.25
        + evidence_quality
        * 0.35
        + decisiveness
        * 0.15
    )


    return clamp(
        confidence
    )


# =========================================================
# SCORE DRIVERS
# =========================================================

def find_score_drivers(
    scores,
):
    report_categories = [
        "growth",
        "profitability",
        "cash_flow",
        "balance_sheet",
        "accounting_quality",
        "peer_strength",
        "multiple_valuation",
        "dcf",
        "valuation_support",
    ]


    category_scores = []


    for category in report_categories:

        score = (
            scores.get(
                category,
                {}
            )
            .get(
                "score"
            )
        )

        if score is None:
            continue


        category_scores.append(
            (
                category,
                score,
            )
        )


    category_scores.sort(
        key=lambda item:
            item[1],
        reverse=True,
    )


    strongest = category_scores[:3]


    weakest = sorted(
        category_scores,
        key=lambda item:
            item[1],
    )[:3]


    return {
        "strongest":
            [
                {
                    "category":
                        category,

                    "score":
                        round(
                            score,
                            1,
                        ),
                }
                for category, score
                in strongest
            ],

        "weakest":
            [
                {
                    "category":
                        category,

                    "score":
                        round(
                            score,
                            1,
                        ),
                }
                for category, score
                in weakest
            ],
    }


# =========================================================
# AI INVESTMENT COMMITTEE
# =========================================================

def generate_ai_explanation(
    research_package,
    peer_analysis,
    recommendation,
):
    if not os.getenv(
        "OPENAI_API_KEY"
    ):

        return None


    client = OpenAI()


    system_prompt = """
You are the explanatory investment-committee layer of a systematic
equity research engine.

The quantitative system has ALREADY determined the research rating.
You MUST NOT change the assigned BUY / HOLD / SELL rating.

Use ONLY the information supplied in the payload.

IMPORTANT DEFINITIONS

Business Quality:
Intrinsic operating quality only. It excludes stock valuation and peer ranking.

Peer Strength:
Relative financial performance versus the supplied comparison group.
It is NOT an economic-moat score and does not prove competitive advantage.

Multiple Valuation:
Absolute and peer-relative market valuation multiples.

DCF:
Scenario-based intrinsic-value analysis. It is not a factual future price.

Valuation Support:
One combined investment-valuation signal incorporating multiple valuation
and DCF. Do not treat Multiple Valuation and DCF as two independent votes.

Rating Confidence:
Strength, coverage, and agreement of evidence.
It is NOT the probability that the rating is correct.

RULES

1. Do not change the assigned rating.
2. Do not invent financial facts.
3. Do not use outside information.
4. Separate company quality from stock attractiveness.
5. Explicitly discuss conflicts in the evidence.
6. Do not claim financial peer strength equals economic moat.
7. Treat DCF outputs as assumption-sensitive scenario estimates.
8. Pay attention to forecast uncertainty and DCF confidence.
9. Mention material accounting flags when present.
10. If peer data is incomplete or inconsistent, say so.
11. Do not turn a HOLD into an implied SELL or BUY in the prose.
12. This is a research rating, not personalized financial advice.

Return:

## Rating Summary

## Business Quality

## Growth and Profitability

## Cash Flow and Balance Sheet

## Accounting Quality

## Peer Comparison

## Valuation

## DCF

## Bull Case

## Bear Case

## Key Risks

## What Would Change the Rating?

## Bottom Line
"""


    payload = {
        "research_package":
            make_serializable(
                research_package
            ),

        "peer_analysis":
            make_serializable(
                peer_analysis
            ),

        "recommendation":
            make_serializable(
                recommendation
            ),
    }


    response = client.responses.create(
        model=MODEL,

        input=[
            {
                "role":
                    "system",

                "content":
                    system_prompt,
            },
            {
                "role":
                    "user",

                "content":
                    json.dumps(
                        payload,
                        indent=2,
                    ),
            },
        ],
    )


    return response.output_text


# =========================================================
# COMPLETE RECOMMENDATION
# =========================================================

def build_recommendation(
    ticker,
    peers=None,
    include_ai=True,
):
    ticker = (
        ticker
        .strip()
        .upper()
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"BUILDING FINAL RECOMMENDATION — {ticker}"
    )

    print(
        "=" * 80
    )


    # -----------------------------------------------------
    # CORE RESEARCH PACKAGE
    # -----------------------------------------------------

    research_package = (
        build_research_package(
            ticker
        )
    )


    # -----------------------------------------------------
    # PEER ANALYSIS
    # -----------------------------------------------------

    print(
        "\nRunning Peer Comparison..."
    )


    peer_analysis = (
        build_peer_analysis(
            ticker,
            peers=peers,
        )
    )


    # -----------------------------------------------------
    # SCORE
    # -----------------------------------------------------

    scores = calculate_scores(
        research_package,
        peer_analysis,
    )


    overall_score = (
        calculate_overall_score(
            scores
        )
    )


    rating_result = determine_rating(
        overall_score,
        scores,
    )


    confidence = (
        calculate_rating_confidence(
            scores,
            overall_score,
        )
    )


    drivers = find_score_drivers(
        scores
    )


    recommendation = {
        "ticker":
            ticker,

        "rating":
            rating_result[
                "rating"
            ],

        "rating_reason":
            rating_result[
                "reason"
            ],

        "overall_score":
            round_or_none(
                overall_score,
                1,
            ),

        "rating_confidence_percent":
            round_or_none(
                confidence,
                1,
            ),

        "business_quality_score":
            round_or_none(
                scores[
                    "business_quality"
                ][
                    "score"
                ],
                1,
            ),

        "peer_strength_score":
            round_or_none(
                scores[
                    "peer_strength"
                ][
                    "score"
                ],
                1,
            ),

        "valuation_support_score":
            round_or_none(
                scores[
                    "valuation_support"
                ][
                    "score"
                ],
                1,
            ),

        "peer_group":
            peer_analysis.get(
                "peer_tickers",
                [],
            ),

        "category_scores":
            {
                category:
                    round_or_none(
                        result.get(
                            "score"
                        ),
                        1,
                    )

                for category, result
                in scores.items()
            },

        "score_details":
            scores,

        "drivers":
            drivers,

        "peer_analysis":
            peer_analysis,

        "research_package":
            research_package,

        "ai_explanation":
            None,
    }


    # -----------------------------------------------------
    # AI EXPLANATION
    # -----------------------------------------------------

    if include_ai:

        try:

            recommendation[
                "ai_explanation"
            ] = generate_ai_explanation(
                research_package,
                peer_analysis,
                recommendation,
            )

        except Exception as error:

            recommendation[
                "ai_explanation"
            ] = (
                "AI explanation unavailable: "
                + str(
                    error
                )
            )


    return recommendation


# =========================================================
# TERMINAL DISPLAY
# =========================================================

def print_recommendation(
    recommendation,
):
    print(
        "\n\n"
        + "=" * 80
    )

    print(
        f"FINAL RESEARCH RATING — {recommendation['ticker']}"
    )

    print(
        "=" * 80
    )


    print(
        f"\nRATING: "
        f"{recommendation['rating']}"
    )

    print(
        f"OVERALL SCORE: "
        f"{recommendation['overall_score']} / 100"
    )

    print(
        f"RATING CONFIDENCE: "
        f"{recommendation['rating_confidence_percent']}%"
    )

    print(
        f"BUSINESS QUALITY: "
        f"{recommendation['business_quality_score']} / 100"
    )

    print(
        f"PEER STRENGTH: "
        f"{recommendation['peer_strength_score']} / 100"
    )

    print(
        f"VALUATION SUPPORT: "
        f"{recommendation['valuation_support_score']} / 100"
    )


    print(
        "\nPEER GROUP:"
    )

    peer_group = recommendation.get(
        "peer_group",
        [],
    )


    if peer_group:

        print(
            ", ".join(
                peer_group
            )
        )

    else:

        print(
            "None"
        )


    print(
        "\nWHY:"
    )

    print(
        recommendation[
            "rating_reason"
        ]
    )


    # =====================================================
    # INTRINSIC BUSINESS SCORES
    # =====================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BUSINESS QUALITY COMPONENTS"
    )

    print(
        "=" * 80
    )


    business_components = {
        "growth":
            "Growth",

        "profitability":
            "Profitability",

        "cash_flow":
            "Cash Flow",

        "balance_sheet":
            "Balance Sheet",

        "accounting_quality":
            "Accounting Quality",
    }


    for key, name in (
        business_components.items()
    ):

        score = (
            recommendation[
                "category_scores"
            ].get(
                key
            )
        )

        print(
            f"{name:<24} "
            f"{score if score is not None else 'N/A'}"
        )


    # =====================================================
    # INVESTMENT SCORES
    # =====================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "INVESTMENT / VALUATION COMPONENTS"
    )

    print(
        "=" * 80
    )


    investment_components = {
        "peer_strength":
            "Peer Strength",

        "multiple_valuation":
            "Multiple Valuation",

        "dcf":
            "DCF Valuation",

        "valuation_support":
            "Valuation Support",

        "model_confidence":
            "Model Confidence",
    }


    for key, name in (
        investment_components.items()
    ):

        score = (
            recommendation[
                "category_scores"
            ].get(
                key
            )
        )

        print(
            f"{name:<24} "
            f"{score if score is not None else 'N/A'}"
        )


    # =====================================================
    # VALUATION BREAKDOWN
    # =====================================================

    valuation_details = (
        recommendation[
            "score_details"
        ][
            "multiple_valuation"
        ]
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "VALUATION BREAKDOWN"
    )

    print(
        "=" * 80
    )


    print(
        "Absolute Valuation Score: "
        f"{round_or_none(valuation_details.get('absolute_score'), 1)}"
    )

    print(
        "Peer-Relative Valuation Score: "
        f"{round_or_none(valuation_details.get('peer_relative_score'), 1)}"
    )


    # =====================================================
    # STRONGEST / WEAKEST
    # =====================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STRONGEST FACTORS"
    )

    print(
        "=" * 80
    )


    for item in (
        recommendation[
            "drivers"
        ][
            "strongest"
        ]
    ):

        print(
            f"{item['category']}: "
            f"{item['score']}"
        )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "WEAKEST FACTORS"
    )

    print(
        "=" * 80
    )


    for item in (
        recommendation[
            "drivers"
        ][
            "weakest"
        ]
    ):

        print(
            f"{item['category']}: "
            f"{item['score']}"
        )


    # =====================================================
    # AI ANALYSIS
    # =====================================================

    explanation = recommendation.get(
        "ai_explanation"
    )


    if explanation:

        print(
            "\n"
            + "=" * 80
        )

        print(
            "INVESTMENT COMMITTEE ANALYSIS"
        )

        print(
            "=" * 80
        )

        print(
            explanation
        )


    # =====================================================
    # DISCLOSURES
    # =====================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "IMPORTANT"
    )

    print(
        "=" * 80
    )


    print(
        "Business Quality measures intrinsic operating quality "
        "and deliberately excludes valuation."
    )

    print(
        "Peer Strength measures financial performance relative "
        "to the supplied comparison group. It is not a moat score."
    )

    print(
        "Valuation Support combines multiple-based valuation "
        "and DCF into one investment-valuation signal so the "
        "same valuation concern is not double-counted."
    )

    print(
        "Rating Confidence describes evidence quality, coverage, "
        "agreement, and model decisiveness. It is not the probability "
        "that the rating is correct."
    )

    print(
        "This is systematic equity research based on financial "
        "evidence and model assumptions, not personalized financial advice."
    )


# =========================================================
# MAIN
# =========================================================

def main():
    ticker = input(
        "Enter stock ticker: "
    ).strip().upper()


    defaults = DEFAULT_PEERS.get(
        ticker,
        [],
    )


    if defaults:

        print(
            "\nDefault comparison group:"
        )

        print(
            ", ".join(
                defaults
            )
        )


    peer_input = input(
        "\nEnter peer tickers separated by commas, "
        "or press Enter to use defaults: "
    ).strip()


    if peer_input:

        peers = [
            item
            .strip()
            .upper()

            for item
            in peer_input.split(",")

            if item.strip()
        ]

    else:

        peers = defaults


    recommendation = build_recommendation(
        ticker,
        peers=peers,
        include_ai=True,
    )


    print_recommendation(
        recommendation
    )


if __name__ == "__main__":
    main()