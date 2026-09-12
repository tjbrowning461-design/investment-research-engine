import sys

from dotenv import load_dotenv

load_dotenv()

import finance_agent

from comparison_engine import (
    get_company_metrics,
)

from dcf_engine import (
    analyze_dcf,
)


# =========================================================
# GENERAL HELPERS
# =========================================================

def format_percent(
    value,
):
    if value is None:
        return None

    try:
        return round(
            float(value) * 100,
            2,
        )

    except Exception:
        return None


def format_number(
    value,
):
    if value is None:
        return None

    try:
        return round(
            float(value),
            2,
        )

    except Exception:
        return None


def make_serializable(
    value,
):
    """
    Convert Pydantic models, pandas/numpy objects,
    dictionaries, and nested structures into
    ordinary Python objects.
    """

    if value is None:
        return None

    if hasattr(
        value,
        "model_dump",
    ):
        return make_serializable(
            value.model_dump()
        )

    if hasattr(
        value,
        "dict",
    ):
        try:
            return make_serializable(
                value.dict()
            )

        except Exception:
            pass

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key):
                make_serializable(item)

            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            make_serializable(item)
            for item
            in value
        ]

    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()

        except Exception:
            pass

    if hasattr(
        value,
        "isoformat",
    ):
        try:
            return value.isoformat()

        except Exception:
            pass

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    return str(value)


# =========================================================
# FINANCIAL BRIEFING AGENT
# =========================================================

def build_financial_briefing_snapshot(
    ticker,
):
    print(
        "\nRunning Financial Briefing Agent..."
    )

    financial_data = (
        finance_agent
        .get_financial_data(
            ticker
        )
    )

    ai_brief = (
        finance_agent
        .generate_ai_brief(
            financial_data
        )
    )

    return {
        "status":
            "loaded",

        "financial_data":
            make_serializable(
                financial_data
            ),

        "executive_brief":
            make_serializable(
                ai_brief
            ),
    }


# =========================================================
# COMPANY COMPARISON SNAPSHOT
# =========================================================

def build_comparison_snapshot(
    ticker,
):
    print(
        "\nRunning Company Comparison Engine..."
    )

    metrics = get_company_metrics(
        ticker
    )

    return {
        "ticker":
            metrics.get(
                "ticker"
            ),

        "company_name":
            metrics.get(
                "company_name"
            ),

        "sector":
            metrics.get(
                "sector"
            ),

        "industry":
            metrics.get(
                "industry"
            ),

        "period_end":
            metrics.get(
                "period_end"
            ),

        "filing_form":
            metrics.get(
                "filing_form"
            ),

        "data_quality":
            metrics.get(
                "data_quality"
            ),

        "revenue_growth_yoy_percent":
            format_percent(
                metrics.get(
                    "revenue_growth_yoy"
                )
            ),

        "operating_income_growth_yoy_percent":
            format_percent(
                metrics.get(
                    "operating_income_growth_yoy"
                )
            ),

        "net_income_growth_yoy_percent":
            format_percent(
                metrics.get(
                    "net_income_growth_yoy"
                )
            ),

        "eps_growth_yoy_percent":
            format_percent(
                metrics.get(
                    "eps_growth_yoy"
                )
            ),

        "gross_margin_percent":
            format_percent(
                metrics.get(
                    "gross_margin"
                )
            ),

        "operating_margin_percent":
            format_percent(
                metrics.get(
                    "operating_margin"
                )
            ),

        "net_margin_percent":
            format_percent(
                metrics.get(
                    "net_margin"
                )
            ),

        "fcf_margin_percent":
            format_percent(
                metrics.get(
                    "fcf_margin"
                )
            ),

        "debt_to_equity":
            format_number(
                metrics.get(
                    "debt_to_equity"
                )
            ),

        "current_ratio":
            format_number(
                metrics.get(
                    "current_ratio"
                )
            ),

        "market_cap":
            metrics.get(
                "market_cap"
            ),

        "trailing_pe":
            format_number(
                metrics.get(
                    "trailing_pe"
                )
            ),

        "forward_pe":
            format_number(
                metrics.get(
                    "forward_pe"
                )
            ),

        "price_to_sales":
            format_number(
                metrics.get(
                    "price_to_sales"
                )
            ),

        "price_to_book":
            format_number(
                metrics.get(
                    "price_to_book"
                )
            ),

        "enterprise_to_ebitda":
            format_number(
                metrics.get(
                    "enterprise_to_ebitda"
                )
            ),

        "enterprise_to_revenue":
            format_number(
                metrics.get(
                    "enterprise_to_revenue"
                )
            ),

        "unusual_non_operating":
            metrics.get(
                "unusual_non_operating"
            ),

        "non_operating_ratio_percent":
            format_percent(
                metrics.get(
                    "non_operating_ratio"
                )
            ),

        "eps_comparability_warning":
            metrics.get(
                "eps_comparability_warning"
            ),

        "missing_metrics":
            metrics.get(
                "missing_metrics",
                [],
            ),
    }


# =========================================================
# DCF SNAPSHOT
# =========================================================

def build_dcf_snapshot(
    ticker,
):
    print(
        "\nRunning DCF Valuation Engine..."
    )

    analysis = analyze_dcf(
        ticker
    )

    base = analysis[
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

    base_scenario = scenarios[
        "Base"
    ]

    return {
        "current_price":
            base.get(
                "current_price"
            ),

        "bear_value":
            scenarios[
                "Bear"
            ][
                "intrinsic_value_per_share"
            ],

        "base_value":
            base_scenario[
                "intrinsic_value_per_share"
            ],

        "bull_value":
            scenarios[
                "Bull"
            ][
                "intrinsic_value_per_share"
            ],

        "valuation_range":
            make_serializable(
                valuation_range
            ),

        "base_upside_downside_percent":
            format_percent(
                base_scenario[
                    "upside_downside"
                ]
            ),

        "base_growth_path_percent":
            [
                format_percent(value)

                for value
                in base_scenario[
                    "assumptions"
                ][
                    "growth_path"
                ]
            ],

        "base_margin_path_percent":
            [
                format_percent(value)

                for value
                in base_scenario[
                    "assumptions"
                ][
                    "margin_path"
                ]
            ],

        "estimated_wacc_percent":
            format_percent(
                base.get(
                    "estimated_wacc"
                )
            ),

        "terminal_growth_percent":
            format_percent(
                base_scenario[
                    "assumptions"
                ][
                    "terminal_growth"
                ]
            ),

        "base_terminal_value_percent":
            format_percent(
                base_scenario[
                    "terminal_value_percent"
                ]
            ),

        "historical_revenue_cagr_percent":
            format_percent(
                base.get(
                    "historical_revenue_cagr"
                )
            ),

        "weighted_recent_growth_percent":
            format_percent(
                base.get(
                    "weighted_recent_growth"
                )
            ),

        "current_operating_margin_percent":
            format_percent(
                base.get(
                    "operating_margin"
                )
            ),

        "normalized_operating_margin_percent":
            format_percent(
                base.get(
                    "normalized_operating_margin"
                )
            ),

        "reverse_dcf_implied_start_growth_percent":
            (
                format_percent(
                    reverse.get(
                        "implied_starting_growth"
                    )
                )

                if reverse

                else None
            ),

        "reverse_dcf":
            make_serializable(
                reverse
            ),

        "data_quality":
            quality.get(
                "data_quality"
            ),

        "historical_growth_stability":
            quality.get(
                "historical_growth_stability"
            ),

        "margin_stability":
            quality.get(
                "margin_stability"
            ),

        "forecast_uncertainty":
            quality.get(
                "forecast_uncertainty"
            ),

        "terminal_value_reliance":
            quality.get(
                "terminal_value_reliance"
            ),

        "overall_dcf_confidence":
            quality.get(
                "overall_confidence"
            ),

        "warnings":
            quality.get(
                "warnings",
                [],
            ),
    }


# =========================================================
# UNIFIED RESEARCH PACKAGE
# =========================================================

def build_research_package(
    ticker,
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
        f"BUILDING UNIFIED RESEARCH PACKAGE — {ticker}"
    )

    print(
        "=" * 80
    )

    package = {
        "ticker":
            ticker,

        "financial_briefing":
            None,

        "comparison":
            None,

        "dcf":
            None,

        "integration_status":
            {},

        "errors":
            [],
    }

    try:
        package[
            "financial_briefing"
        ] = build_financial_briefing_snapshot(
            ticker
        )

        package[
            "integration_status"
        ][
            "financial_briefing_agent"
        ] = "success"

    except Exception as error:
        package[
            "integration_status"
        ][
            "financial_briefing_agent"
        ] = "failed"

        package[
            "errors"
        ].append(
            {
                "source":
                    "financial_briefing_agent",

                "error":
                    str(error),
            }
        )

    try:
        package[
            "comparison"
        ] = build_comparison_snapshot(
            ticker
        )

        package[
            "integration_status"
        ][
            "comparison_engine"
        ] = "success"

    except Exception as error:
        package[
            "integration_status"
        ][
            "comparison_engine"
        ] = "failed"

        package[
            "errors"
        ].append(
            {
                "source":
                    "comparison_engine",

                "error":
                    str(error),
            }
        )

    try:
        package[
            "dcf"
        ] = build_dcf_snapshot(
            ticker
        )

        package[
            "integration_status"
        ][
            "dcf_engine"
        ] = "success"

    except Exception as error:
        package[
            "integration_status"
        ][
            "dcf_engine"
        ] = "failed"

        package[
            "errors"
        ].append(
            {
                "source":
                    "dcf_engine",

                "error":
                    str(error),
            }
        )

    return package


# =========================================================
# TERMINAL DISPLAY
# =========================================================

def print_research_package(
    package,
):
    print(
        "\n\n"
        + "=" * 80
    )

    print(
        f"INVESTMENT RESEARCH ENGINE — {package['ticker']}"
    )

    print(
        "=" * 80
    )

    print(
        "\nINTEGRATION STATUS"
    )

    print(
        "-" * 80
    )

    for engine, status in (
        package[
            "integration_status"
        ].items()
    ):
        print(
            f"{engine}: {status}"
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINANCIAL BRIEFING AGENT"
    )

    print(
        "=" * 80
    )

    briefing = package.get(
        "financial_briefing"
    )

    if briefing:
        print(
            f"status: {briefing.get('status')}"
        )

        executive_brief = briefing.get(
            "executive_brief"
        )

        if executive_brief:
            print(
                "\nExecutive Summary:"
            )

            summary = (
                executive_brief.get(
                    "executive_summary"
                )

                if isinstance(
                    executive_brief,
                    dict,
                )

                else None
            )

            if summary:
                print(summary)

            else:
                print(
                    "Structured AI brief loaded successfully."
                )

        else:
            print(
                "AI brief unavailable."
            )

    else:
        print(
            "Unavailable"
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "COMPANY / FUNDAMENTAL SNAPSHOT"
    )

    print(
        "=" * 80
    )

    comparison = package.get(
        "comparison"
    )

    if comparison:
        key_metrics = [
            "company_name",
            "period_end",
            "data_quality",
            "revenue_growth_yoy_percent",
            "operating_income_growth_yoy_percent",
            "net_income_growth_yoy_percent",
            "eps_growth_yoy_percent",
            "gross_margin_percent",
            "operating_margin_percent",
            "net_margin_percent",
            "fcf_margin_percent",
            "debt_to_equity",
            "current_ratio",
            "forward_pe",
            "price_to_sales",
            "enterprise_to_ebitda",
            "unusual_non_operating",
        ]

        for key in key_metrics:
            print(
                f"{key}: "
                f"{comparison.get(key)}"
            )

    else:
        print(
            "Unavailable"
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DCF VALUATION"
    )

    print(
        "=" * 80
    )

    dcf = package.get(
        "dcf"
    )

    if dcf:
        print(
            f"Current Price: "
            f"{dcf.get('current_price')}"
        )

        print(
            f"Bear Value: "
            f"{dcf.get('bear_value')}"
        )

        print(
            f"Base Value: "
            f"{dcf.get('base_value')}"
        )

        print(
            f"Bull Value: "
            f"{dcf.get('bull_value')}"
        )

        print(
            f"Base Upside / Downside: "
            f"{dcf.get('base_upside_downside_percent')}%"
        )

        print(
            f"Estimated WACC: "
            f"{dcf.get('estimated_wacc_percent')}%"
        )

        print(
            f"Reverse DCF Implied Starting Growth: "
            f"{dcf.get('reverse_dcf_implied_start_growth_percent')}%"
        )

        print(
            f"DCF Confidence: "
            f"{dcf.get('overall_dcf_confidence')}"
        )

        print(
            f"Forecast Uncertainty: "
            f"{dcf.get('forecast_uncertainty')}"
        )

    else:
        print(
            "Unavailable"
        )

    if package[
        "errors"
    ]:
        print(
            "\n"
            + "=" * 80
        )

        print(
            "INTEGRATION ERRORS"
        )

        print(
            "=" * 80
        )

        for error in package[
            "errors"
        ]:
            print(
                f"{error['source']}: "
                f"{error['error']}"
            )


# =========================================================
# MAIN
# =========================================================

def main():
    ticker = input(
        "Enter stock ticker: "
    )

    package = build_research_package(
        ticker
    )

    print_research_package(
        package
    )


if __name__ == "__main__":
    main()