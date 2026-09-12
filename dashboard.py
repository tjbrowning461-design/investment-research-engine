import pandas as pd
import altair as alt
import streamlit as st

from recommendation_engine import (
    DEFAULT_PEERS,
    build_recommendation,
)


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Investment Research Engine",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .main-title {
        font-size: 2.7rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        color: #777;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.20);
        border-radius: 12px;
        padding: 14px;
    }

    .rating-box {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 18px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================

def percent(value):
    if value is None:
        return "N/A"

    return f"{value:,.2f}%"


def money(value):
    if value is None:
        return "N/A"

    value = float(value)

    absolute = abs(value)

    if absolute >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:,.2f}T"

    if absolute >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"

    if absolute >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"

    return f"${value:,.2f}"


def score(value):
    if value is None:
        return "N/A"

    return f"{value:,.1f} / 100"


def share_price(value):
    if value is None:
        return "N/A"

    return f"${float(value):,.2f}"


# =========================================================
# CHART HELPERS
# =========================================================

def category_score_chart(category_scores):

    rows = []

    labels = {
        "growth": "Growth",
        "profitability": "Profitability",
        "cash_flow": "Cash Flow",
        "balance_sheet": "Balance Sheet",
        "accounting_quality": "Accounting Quality",
        "peer_strength": "Peer Strength",
        "multiple_valuation": "Multiple Valuation",
        "dcf": "DCF Valuation",
        "valuation_support": "Valuation Support",
        "model_confidence": "Model Confidence",
    }

    for key, label in labels.items():

        value = category_scores.get(
            key
        )

        if value is None:
            continue

        rows.append(
            {
                "Category": label,
                "Score": value,
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    return (
        alt.Chart(
            dataframe
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "Score:Q",
                scale=alt.Scale(
                    domain=[
                        0,
                        100,
                    ]
                ),
                title="Score",
            ),
            y=alt.Y(
                "Category:N",
                sort="-x",
                title=None,
            ),
            tooltip=[
                "Category:N",
                alt.Tooltip(
                    "Score:Q",
                    format=".1f",
                ),
            ],
        )
        .properties(
            height=380
        )
    )


def dcf_chart(
    dcf,
):

    rows = [
        {
            "Scenario": "Bear",
            "Value": dcf.get(
                "bear_value"
            ),
        },
        {
            "Scenario": "Base",
            "Value": dcf.get(
                "base_value"
            ),
        },
        {
            "Scenario": "Bull",
            "Value": dcf.get(
                "bull_value"
            ),
        },
    ]

    dataframe = pd.DataFrame(
        rows
    )

    bars = (
        alt.Chart(
            dataframe
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "Scenario:N",
                title=None,
            ),
            y=alt.Y(
                "Value:Q",
                title="Value per Share ($)",
            ),
            tooltip=[
                "Scenario:N",
                alt.Tooltip(
                    "Value:Q",
                    format="$,.2f",
                ),
            ],
        )
    )

    current_price = dcf.get(
        "current_price"
    )

    if current_price is None:

        return bars.properties(
            height=350
        )

    price_data = pd.DataFrame(
        {
            "Current Price": [
                current_price
            ]
        }
    )

    rule = (
        alt.Chart(
            price_data
        )
        .mark_rule(
            strokeDash=[
                6,
                4,
            ]
        )
        .encode(
            y="Current Price:Q"
        )
    )

    return (
        bars
        + rule
    ).properties(
        height=350
    )


def peer_table_from_analysis(
    peer_analysis,
):

    target = peer_analysis.get(
        "target"
    )

    peers = peer_analysis.get(
        "peers",
        [],
    )

    companies = []

    if target:
        companies.append(
            target
        )

    companies.extend(
        peers
    )

    rows = []

    for company in companies:

        rows.append(
            {
                "Ticker":
                    company.get(
                        "ticker"
                    ),

                "Revenue Growth":
                    company.get(
                        "revenue_growth_yoy"
                    ),

                "Operating Growth":
                    company.get(
                        "operating_income_growth_yoy"
                    ),

                "Gross Margin":
                    company.get(
                        "gross_margin"
                    ),

                "Operating Margin":
                    company.get(
                        "operating_margin"
                    ),

                "FCF Margin":
                    company.get(
                        "fcf_margin"
                    ),

                "Current Ratio":
                    company.get(
                        "current_ratio"
                    ),

                "Debt / Equity":
                    company.get(
                        "debt_to_equity"
                    ),

                "Forward P/E":
                    company.get(
                        "forward_pe"
                    ),

                "EV / EBITDA":
                    company.get(
                        "enterprise_to_ebitda"
                    ),

                "Data Quality":
                    company.get(
                        "data_quality"
                    ),
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    percent_columns = [
        "Revenue Growth",
        "Operating Growth",
        "Gross Margin",
        "Operating Margin",
        "FCF Margin",
    ]

    for column in percent_columns:

        if column in dataframe.columns:

            dataframe[
                column
            ] = (
                dataframe[
                    column
                ]
                * 100
            )

    return dataframe


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">📊 Investment Research Engine</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Unified fundamental analysis, peer comparison, DCF valuation,
    systematic Buy / Hold / Sell research rating, and AI investment-committee analysis.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header(
    "Research Setup"
)

ticker = st.sidebar.text_input(
    "Ticker",
    value="AAPL",
).strip().upper()


default_peers = DEFAULT_PEERS.get(
    ticker,
    [],
)


peer_default_text = ", ".join(
    default_peers
)


peer_input = st.sidebar.text_area(
    "Peer group",
    value=peer_default_text,
    height=110,
)


include_ai = st.sidebar.checkbox(
    "Generate AI investment-committee analysis",
    value=True,
)


run_button = st.sidebar.button(
    "Run Full Research",
    type="primary",
    use_container_width=True,
)


st.sidebar.divider()

st.sidebar.caption(
    "Peer groups are research comparison sets and are not necessarily direct competitors."
)

st.sidebar.caption(
    "Research ratings are systematic model outputs, not personalized financial advice."
)


# =========================================================
# RUN
# =========================================================

if run_button:

    if not ticker:

        st.sidebar.error(
            "Enter a ticker."
        )

        st.stop()


    peers = [
        item.strip().upper()
        for item in peer_input.split(",")
        if item.strip()
    ]


    with st.spinner(
        f"Running full investment research for {ticker}..."
    ):

        try:

            recommendation = build_recommendation(
                ticker,
                peers=peers,
                include_ai=include_ai,
            )


            st.session_state[
                "recommendation"
            ] = recommendation


        except Exception as error:

            st.error(
                f"Research engine failed: {error}"
            )

            st.stop()


# =========================================================
# EMPTY STATE
# =========================================================

if "recommendation" not in st.session_state:

    st.info(
        "Enter a ticker and click **Run Full Research**."
    )

    st.stop()


# =========================================================
# LOAD RESULT
# =========================================================

recommendation = st.session_state[
    "recommendation"
]


research_package = recommendation[
    "research_package"
]


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


peer_analysis = recommendation[
    "peer_analysis"
]


category_scores = recommendation[
    "category_scores"
]


# =========================================================
# COMPANY HEADER
# =========================================================

company_name = comparison.get(
    "company_name",
    ticker,
)


st.divider()

st.markdown(
    f"## {company_name} ({recommendation['ticker']})"
)


# =========================================================
# FINAL RATING
# =========================================================

st.markdown(
    f"""
    <div class="rating-box">
        <h2 style="margin-bottom:0.2rem;">
            Research Rating: {recommendation['rating']}
        </h2>
        <p style="margin-top:0;">
            {recommendation['rating_reason']}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


col1, col2, col3, col4, col5 = st.columns(
    5
)


col1.metric(
    "Overall Score",
    score(
        recommendation[
            "overall_score"
        ]
    ),
)


col2.metric(
    "Rating Confidence",
    percent(
        recommendation[
            "rating_confidence_percent"
        ]
    ),
)


col3.metric(
    "Business Quality",
    score(
        recommendation[
            "business_quality_score"
        ]
    ),
)


col4.metric(
    "Peer Strength",
    score(
        recommendation[
            "peer_strength_score"
        ]
    ),
)


col5.metric(
    "Valuation Support",
    score(
        recommendation[
            "valuation_support_score"
        ]
    ),
)


# =========================================================
# TABS
# =========================================================

(
    overview_tab,
    business_tab,
    peer_tab,
    valuation_tab,
    dcf_tab,
    drivers_tab,
    ai_tab,
    methodology_tab,
) = st.tabs(
    [
        "Overview",
        "Business Quality",
        "Peer Comparison",
        "Valuation",
        "DCF",
        "Drivers",
        "AI Committee",
        "Methodology",
    ]
)


# =========================================================
# OVERVIEW
# =========================================================

with overview_tab:

    st.subheader(
        "Research Scorecard"
    )


    st.altair_chart(
        category_score_chart(
            category_scores
        ),
        use_container_width=True,
    )


    st.subheader(
        "Key Company Metrics"
    )


    col1, col2, col3, col4 = st.columns(
        4
    )


    col1.metric(
        "Revenue Growth YoY",
        percent(
            comparison.get(
                "revenue_growth_yoy_percent"
            )
        ),
    )


    col2.metric(
        "Operating Margin",
        percent(
            comparison.get(
                "operating_margin_percent"
            )
        ),
    )


    col3.metric(
        "FCF Margin",
        percent(
            comparison.get(
                "fcf_margin_percent"
            )
        ),
    )


    col4.metric(
        "Forward P/E",
        (
            f"{comparison.get('forward_pe'):,.2f}x"
            if comparison.get(
                "forward_pe"
            )
            is not None
            else "N/A"
        ),
    )


    col1, col2, col3, col4 = st.columns(
        4
    )


    col1.metric(
        "Current Price",
        share_price(
            dcf.get(
                "current_price"
            )
        ),
    )


    col2.metric(
        "DCF Base Value",
        share_price(
            dcf.get(
                "base_value"
            )
        ),
    )


    col3.metric(
        "DCF Bull Value",
        share_price(
            dcf.get(
                "bull_value"
            )
        ),
    )


    col4.metric(
        "Reverse DCF Growth",
        percent(
            dcf.get(
                "reverse_dcf_implied_start_growth_percent"
            )
        ),
    )


# =========================================================
# BUSINESS QUALITY
# =========================================================

with business_tab:

    st.subheader(
        "Intrinsic Business Quality"
    )


    st.caption(
        "This section deliberately excludes valuation and peer ranking."
    )


    business_components = {
        "Growth":
            category_scores.get(
                "growth"
            ),

        "Profitability":
            category_scores.get(
                "profitability"
            ),

        "Cash Flow":
            category_scores.get(
                "cash_flow"
            ),

        "Balance Sheet":
            category_scores.get(
                "balance_sheet"
            ),

        "Accounting Quality":
            category_scores.get(
                "accounting_quality"
            ),
    }


    rows = [
        {
            "Category":
                name,

            "Score":
                value,
        }

        for name, value
        in business_components.items()
    ]


    st.dataframe(
        pd.DataFrame(
            rows
        ),
        use_container_width=True,
        hide_index=True,
    )


    st.subheader(
        "Fundamental Snapshot"
    )


    fundamental_rows = [
        {
            "Metric":
                "Revenue Growth YoY",

            "Value":
                percent(
                    comparison.get(
                        "revenue_growth_yoy_percent"
                    )
                ),
        },
        {
            "Metric":
                "Operating Income Growth YoY",

            "Value":
                percent(
                    comparison.get(
                        "operating_income_growth_yoy_percent"
                    )
                ),
        },
        {
            "Metric":
                "Net Income Growth YoY",

            "Value":
                percent(
                    comparison.get(
                        "net_income_growth_yoy_percent"
                    )
                ),
        },
        {
            "Metric":
                "EPS Growth YoY",

            "Value":
                percent(
                    comparison.get(
                        "eps_growth_yoy_percent"
                    )
                ),
        },
        {
            "Metric":
                "Gross Margin",

            "Value":
                percent(
                    comparison.get(
                        "gross_margin_percent"
                    )
                ),
        },
        {
            "Metric":
                "Operating Margin",

            "Value":
                percent(
                    comparison.get(
                        "operating_margin_percent"
                    )
                ),
        },
        {
            "Metric":
                "Net Margin",

            "Value":
                percent(
                    comparison.get(
                        "net_margin_percent"
                    )
                ),
        },
        {
            "Metric":
                "FCF Margin",

            "Value":
                percent(
                    comparison.get(
                        "fcf_margin_percent"
                    )
                ),
        },
        {
            "Metric":
                "Debt / Equity",

            "Value":
                comparison.get(
                    "debt_to_equity"
                ),
        },
        {
            "Metric":
                "Current Ratio",

            "Value":
                comparison.get(
                    "current_ratio"
                ),
        },
    ]


    st.dataframe(
        pd.DataFrame(
            fundamental_rows
        ),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# PEER COMPARISON
# =========================================================

with peer_tab:

    st.subheader(
        "Peer Comparison"
    )


    st.write(
        "Comparison group:"
    )


    st.code(
        ", ".join(
            recommendation.get(
                "peer_group",
                [],
            )
        )
    )


    peer_dataframe = (
        peer_table_from_analysis(
            peer_analysis
        )
    )


    st.dataframe(
        peer_dataframe,
        use_container_width=True,
        hide_index=True,
    )


    st.metric(
        "Peer Strength Score",
        score(
            recommendation[
                "peer_strength_score"
            ]
        ),
    )


    st.info(
        "Peer Strength measures relative financial performance within this comparison set. "
        "It is not a direct measure of economic moat or competitive advantage."
    )


# =========================================================
# VALUATION
# =========================================================

with valuation_tab:

    st.subheader(
        "Valuation Support"
    )


    st.metric(
        "Combined Valuation Support",
        score(
            recommendation[
                "valuation_support_score"
            ]
        ),
    )


    valuation_details = (
        recommendation[
            "score_details"
        ][
            "multiple_valuation"
        ]
    )


    col1, col2, col3 = st.columns(
        3
    )


    col1.metric(
        "Absolute Valuation Score",
        score(
            valuation_details.get(
                "absolute_score"
            )
        ),
    )


    col2.metric(
        "Peer-Relative Valuation Score",
        score(
            valuation_details.get(
                "peer_relative_score"
            )
        ),
    )


    col3.metric(
        "DCF Score",
        score(
            category_scores.get(
                "dcf"
            )
        ),
    )


    multiple_rows = [
        {
            "Metric":
                "Trailing P/E",

            "Value":
                comparison.get(
                    "trailing_pe"
                ),
        },
        {
            "Metric":
                "Forward P/E",

            "Value":
                comparison.get(
                    "forward_pe"
                ),
        },
        {
            "Metric":
                "Price / Sales",

            "Value":
                comparison.get(
                    "price_to_sales"
                ),
        },
        {
            "Metric":
                "Price / Book",

            "Value":
                comparison.get(
                    "price_to_book"
                ),
        },
        {
            "Metric":
                "EV / EBITDA",

            "Value":
                comparison.get(
                    "enterprise_to_ebitda"
                ),
        },
        {
            "Metric":
                "EV / Revenue",

            "Value":
                comparison.get(
                    "enterprise_to_revenue"
                ),
        },
    ]


    st.dataframe(
        pd.DataFrame(
            multiple_rows
        ),
        use_container_width=True,
        hide_index=True,
    )


    st.caption(
        "Valuation Support combines multiple-based valuation and DCF into one signal "
        "so the same expensive/cheap stock concern is not double-counted."
    )


# =========================================================
# DCF TAB
# =========================================================

with dcf_tab:

    st.subheader(
        "DCF Valuation"
    )


    st.altair_chart(
        dcf_chart(
            dcf
        ),
        use_container_width=True,
    )


    col1, col2, col3, col4 = st.columns(
        4
    )


    col1.metric(
        "Bear Value",
        share_price(
            dcf.get(
                "bear_value"
            )
        ),
    )


    col2.metric(
        "Base Value",
        share_price(
            dcf.get(
                "base_value"
            )
        ),
    )


    col3.metric(
        "Bull Value",
        share_price(
            dcf.get(
                "bull_value"
            )
        ),
    )


    col4.metric(
        "Current Price",
        share_price(
            dcf.get(
                "current_price"
            )
        ),
    )


    col1, col2, col3, col4 = st.columns(
        4
    )


    col1.metric(
        "Base Upside / Downside",
        percent(
            dcf.get(
                "base_upside_downside_percent"
            )
        ),
    )


    col2.metric(
        "WACC",
        percent(
            dcf.get(
                "estimated_wacc_percent"
            )
        ),
    )


    col3.metric(
        "Reverse DCF Growth",
        percent(
            dcf.get(
                "reverse_dcf_implied_start_growth_percent"
            )
        ),
    )


    col4.metric(
        "DCF Confidence",
        dcf.get(
            "overall_dcf_confidence",
            "N/A",
        ),
    )


    if dcf.get(
        "warnings"
    ):

        st.subheader(
            "DCF Warnings"
        )

        for warning in dcf[
            "warnings"
        ]:

            st.warning(
                warning
            )


# =========================================================
# DRIVERS
# =========================================================

with drivers_tab:

    st.subheader(
        "Strongest Factors"
    )


    strongest = recommendation[
        "drivers"
    ][
        "strongest"
    ]


    strongest_df = pd.DataFrame(
        strongest
    )


    st.dataframe(
        strongest_df,
        use_container_width=True,
        hide_index=True,
    )


    st.subheader(
        "Weakest Factors"
    )


    weakest = recommendation[
        "drivers"
    ][
        "weakest"
    ]


    weakest_df = pd.DataFrame(
        weakest
    )


    st.dataframe(
        weakest_df,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# AI COMMITTEE
# =========================================================

with ai_tab:

    st.subheader(
        "Investment Committee Analysis"
    )


    explanation = recommendation.get(
        "ai_explanation"
    )


    if explanation:

        st.markdown(
            explanation
        )

    else:

        st.info(
            "AI investment-committee analysis was not generated."
        )


# =========================================================
# METHODOLOGY
# =========================================================

with methodology_tab:

    st.subheader(
        "How the Rating Works"
    )


    st.markdown(
        """
### Business Quality

Measures intrinsic operating quality using:

- Growth
- Profitability
- Cash flow
- Balance-sheet strength
- Accounting quality

It deliberately excludes valuation.

### Peer Strength

Measures financial performance relative to the supplied comparison group.

It does **not** claim to measure economic moat or competitive advantage.

### Valuation Support

Combines:

- Absolute valuation multiples
- Peer-relative valuation
- Confidence-adjusted DCF valuation

DCF and market multiples are therefore treated as parts of one valuation judgment,
rather than independent votes that can double-count the same signal.

### Final Rating

The deterministic scoring engine assigns:

- **BUY**
- **HOLD**
- **SELL**

The AI explanation cannot override the algorithmic rating.

### Rating Confidence

Rating Confidence measures:

- Evidence coverage
- Agreement between model components
- DCF/model quality
- How decisive the overall score is

It is **not** the probability that the rating is correct.

### Important Limitation

The system is a research framework based on financial evidence and explicit model assumptions.
It is not personalized financial advice and cannot guarantee future investment performance.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Investment Research Engine • Fundamental Analysis • Peer Comparison • "
    "DCF Valuation • Systematic Research Rating • AI Investment Committee"
)