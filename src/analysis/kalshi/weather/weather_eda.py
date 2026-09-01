"""Exploratory data analysis of Kalshi weather (high/low temp) markets.

Isolates weather temperature markets and examines:
- Volume by city and market type (high vs low)
- Trade count distribution over time of day
- Average trade size patterns
- Win rate calibration for weather markets specifically
- Price distribution at time of trade
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType

# Kalshi weather ticker prefixes for high/low temp markets.
# These match event_ticker patterns like HIGHNY-25FEB14, LOWCHI-25FEB14, etc.
WEATHER_TEMP_PREFIXES = (
    "HIGH",  # High temp markets (all cities)
    "LOW",  # Low temp markets (all cities)
)

# Map ticker prefix → city for known stations
CITY_MAP = {
    "NY": "New York",
    "CHI": "Chicago",
    "AUS": "Austin",
    "MIA": "Miami",
    "LAX": "Los Angeles",
    "DEN": "Denver",
    "PHIL": "Philadelphia",
    "HOU": "Houston",
    "DFW": "Dallas",
    "MSY": "New Orleans",
    "LGA": "LaGuardia",
    "DAL": "Dallas Love",
    "SEA": "Seattle",
    "SFO": "San Francisco",
    "LAS": "Las Vegas",
    "PHX": "Phoenix",
    "SAT": "San Antonio",
    "DCA": "Washington DC",
    "CLT": "Charlotte",
    "BOS": "Boston",
    "BNA": "Nashville",
    "ATL": "Atlanta",
    "JAX": "Jacksonville",
    "OKC": "Oklahoma City",
    "DTW": "Detroit",
    "MSP": "Minneapolis",
}


def _weather_filter_sql() -> str:
    """SQL WHERE clause to filter for weather high/low temp markets."""
    return """
    (
        event_ticker LIKE 'HIGH%'
        OR event_ticker LIKE 'LOW%'
    )
    AND event_ticker NOT LIKE 'HIGHLIGHT%'
    AND event_ticker NOT LIKE 'LOWBALL%'
    """


class WeatherEdaAnalysis(Analysis):
    """Exploratory data analysis of Kalshi weather temperature markets."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="weather_eda",
            description="EDA of Kalshi weather high/low temperature markets",
        )
        base_dir = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()
        weather_filter = _weather_filter_sql()

        # --- Panel 1: Volume by city ---
        df_city = con.execute(
            f"""
            WITH weather_markets AS (
                SELECT ticker, event_ticker, result
                FROM '{self.markets_dir}/*.parquet'
                WHERE {weather_filter}
            ),
            weather_trades AS (
                SELECT
                    t.*,
                    m.event_ticker,
                    m.result
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN weather_markets m ON t.ticker = m.ticker
            )
            SELECT
                -- Extract city code from event_ticker (e.g. HIGHNY -> NY, LOWCHI -> CHI)
                CASE
                    WHEN event_ticker LIKE 'HIGH%' THEN
                        regexp_extract(event_ticker, '^HIGH([A-Z]+)', 1)
                    WHEN event_ticker LIKE 'LOW%' THEN
                        regexp_extract(event_ticker, '^LOW([A-Z]+)', 1)
                END AS city_code,
                CASE
                    WHEN event_ticker LIKE 'HIGH%' THEN 'High'
                    WHEN event_ticker LIKE 'LOW%' THEN 'Low'
                END AS market_type,
                COUNT(*) AS n_trades,
                SUM(t.count) AS total_contracts,
                SUM(t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                               ELSE t.no_price END) / 100.0) AS volume_usd
            FROM weather_trades t
            GROUP BY city_code, market_type
            ORDER BY volume_usd DESC
            """
        ).df()

        # --- Panel 2: Trades by hour of day (UTC) ---
        df_hourly = con.execute(
            f"""
            WITH weather_markets AS (
                SELECT ticker, event_ticker
                FROM '{self.markets_dir}/*.parquet'
                WHERE {weather_filter}
            )
            SELECT
                EXTRACT(HOUR FROM t.created_time) AS hour_utc,
                CASE
                    WHEN m.event_ticker LIKE 'HIGH%' THEN 'High'
                    WHEN m.event_ticker LIKE 'LOW%' THEN 'Low'
                END AS market_type,
                COUNT(*) AS n_trades,
                SUM(t.count) AS total_contracts,
                AVG(t.count) AS avg_trade_size
            FROM '{self.trades_dir}/*.parquet' t
            INNER JOIN weather_markets m ON t.ticker = m.ticker
            GROUP BY hour_utc, market_type
            ORDER BY hour_utc
            """
        ).df()

        # --- Panel 3: Price distribution at time of trade ---
        df_price = con.execute(
            f"""
            WITH weather_markets AS (
                SELECT ticker, event_ticker, result
                FROM '{self.markets_dir}/*.parquet'
                WHERE {weather_filter}
                  AND status = 'finalized'
                  AND result IN ('yes', 'no')
            )
            SELECT
                CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won,
                CASE
                    WHEN m.event_ticker LIKE 'HIGH%' THEN 'High'
                    WHEN m.event_ticker LIKE 'LOW%' THEN 'Low'
                END AS market_type
            FROM '{self.trades_dir}/*.parquet' t
            INNER JOIN weather_markets m ON t.ticker = m.ticker
            """
        ).df()

        # --- Panel 4: Daily volume over time ---
        df_daily = con.execute(
            f"""
            WITH weather_markets AS (
                SELECT ticker, event_ticker
                FROM '{self.markets_dir}/*.parquet'
                WHERE {weather_filter}
            )
            SELECT
                DATE_TRUNC('week', t.created_time) AS week,
                COUNT(*) AS n_trades,
                SUM(t.count) AS total_contracts,
                SUM(t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                               ELSE t.no_price END) / 100.0) AS volume_usd
            FROM '{self.trades_dir}/*.parquet' t
            INNER JOIN weather_markets m ON t.ticker = m.ticker
            GROUP BY week
            ORDER BY week
            """
        ).df()

        # --- Panel 5: Summary stats ---
        df_summary = con.execute(
            f"""
            WITH weather_markets AS (
                SELECT ticker, event_ticker, result, status
                FROM '{self.markets_dir}/*.parquet'
                WHERE {weather_filter}
            )
            SELECT
                COUNT(DISTINCT m.ticker) AS n_markets,
                COUNT(*) AS n_trades,
                SUM(t.count) AS total_contracts,
                SUM(t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                               ELSE t.no_price END) / 100.0) AS total_volume_usd,
                AVG(t.count) AS avg_trade_size,
                MIN(t.created_time) AS first_trade,
                MAX(t.created_time) AS last_trade
            FROM '{self.trades_dir}/*.parquet' t
            INNER JOIN weather_markets m ON t.ticker = m.ticker
            """
        ).df()

        fig = self._create_figure(df_city, df_hourly, df_price, df_daily, df_summary)
        chart = self._create_chart(df_hourly)

        # Combine all dataframes into one output
        combined_data = pd.concat(
            [
                df_summary.assign(panel="summary"),
                df_city.assign(panel="city_volume"),
                df_hourly.assign(panel="hourly_trades"),
            ],
            ignore_index=True,
        )

        return AnalysisOutput(
            figure=fig,
            data=combined_data,
            chart=chart,
            metadata={
                "total_markets": (
                    int(df_summary["n_markets"].iloc[0]) if len(df_summary) else 0
                ),
                "total_trades": (
                    int(df_summary["n_trades"].iloc[0]) if len(df_summary) else 0
                ),
            },
        )

    def _create_figure(
        self,
        df_city: pd.DataFrame,
        df_hourly: pd.DataFrame,
        df_price: pd.DataFrame,
        df_daily: pd.DataFrame,
        df_summary: pd.DataFrame,
    ) -> plt.Figure:
        """Create 2x2 EDA figure."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Kalshi Weather Temperature Markets — EDA", fontsize=16, fontweight="bold"
        )

        # --- Top-left: Volume by city (top 15) ---
        ax = axes[0, 0]
        if len(df_city) > 0:
            city_agg = (
                df_city.groupby("city_code")["volume_usd"]
                .sum()
                .sort_values(ascending=True)
                .tail(15)
            )
            colors = [
                "#17becf" if code in ("NY", "CHI", "MIA", "LAX", "DEN") else "#a0d8e8"
                for code in city_agg.index
            ]
            city_agg.plot.barh(ax=ax, color=colors)
            ax.set_xlabel("Total Volume (USD)")
            ax.set_title("Volume by City (Top 15)")
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda x, _: f"${x/1e3:.0f}K" if x < 1e6 else f"${x/1e6:.1f}M"
                )
            )
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title("Volume by City")

        # --- Top-right: Trades by hour of day ---
        ax = axes[0, 1]
        if len(df_hourly) > 0:
            for mtype, color in [("High", "#d62728"), ("Low", "#1f77b4")]:
                subset = df_hourly[df_hourly["market_type"] == mtype]
                if len(subset) > 0:
                    ax.bar(
                        subset["hour_utc"] + (0.2 if mtype == "High" else -0.2),
                        subset["n_trades"],
                        width=0.4,
                        alpha=0.7,
                        label=mtype,
                        color=color,
                    )
            ax.set_xlabel("Hour of Day (UTC)")
            ax.set_ylabel("Number of Trades")
            ax.set_title("Trade Count by Hour (UTC)")
            ax.set_xlim(-0.5, 23.5)
            ax.set_xticks(range(0, 24, 2))
            ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

            # Annotate METAR observation windows
            # Standard METAR: typically :53 each hour; DSM/CLI releases vary
            for hour in [6, 12, 18]:  # Key UTC observation hours
                ax.axvline(x=hour, color="gray", linestyle=":", alpha=0.5)
            ax.annotate(
                "METAR obs\ntypically :53",
                xy=(12, ax.get_ylim()[1] * 0.9),
                fontsize=7,
                color="gray",
                ha="center",
            )
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title("Trade Count by Hour")

        # --- Bottom-left: Win rate calibration for weather ---
        ax = axes[1, 0]
        if len(df_price) > 0:
            for mtype, color, marker in [
                ("High", "#d62728", "o"),
                ("Low", "#1f77b4", "s"),
            ]:
                subset = df_price[df_price["market_type"] == mtype]
                if len(subset) > 0:
                    binned = (
                        subset.groupby(subset["price"] // 5 * 5)
                        .agg(
                            win_rate=("won", "mean"),
                            n=("won", "count"),
                        )
                        .reset_index()
                    )
                    binned = binned[binned["n"] >= 10]  # Filter sparse bins
                    ax.scatter(
                        binned["price"],
                        binned["win_rate"] * 100,
                        s=np.sqrt(binned["n"]) * 3,
                        alpha=0.7,
                        label=mtype,
                        color=color,
                        marker=marker,
                    )
            ax.plot(
                [0, 100],
                [0, 100],
                "--",
                color="gray",
                alpha=0.5,
                label="Perfect calibration",
            )
            ax.set_xlabel("Contract Price (cents)")
            ax.set_ylabel("Win Rate (%)")
            ax.set_title("Weather Market Calibration")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_aspect("equal")
            ax.legend(loc="upper left", fontsize=8)
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title("Weather Market Calibration")

        # --- Bottom-right: Weekly volume over time ---
        ax = axes[1, 1]
        if len(df_daily) > 0:
            ax.fill_between(
                df_daily["week"],
                df_daily["volume_usd"] / 1e3,
                alpha=0.3,
                color="#17becf",
            )
            ax.plot(
                df_daily["week"],
                df_daily["volume_usd"] / 1e3,
                color="#17becf",
                linewidth=1.5,
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Weekly Volume ($K)")
            ax.set_title("Weather Market Volume Over Time")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title("Weather Market Volume Over Time")

        # Add summary text
        if len(df_summary) > 0 and df_summary["n_trades"].iloc[0] > 0:
            s = df_summary.iloc[0]
            summary_text = (
                f"Markets: {int(s['n_markets']):,}  |  "
                f"Trades: {int(s['n_trades']):,}  |  "
                f"Contracts: {int(s['total_contracts']):,}  |  "
                f"Volume: ${s['total_volume_usd']:,.0f}"
            )
            fig.text(
                0.5,
                0.02,
                summary_text,
                ha="center",
                fontsize=10,
                style="italic",
                color="gray",
            )

        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        return fig

    def _create_chart(self, df_hourly: pd.DataFrame) -> ChartConfig:
        """Create chart config for the hourly trade distribution."""
        if len(df_hourly) == 0:
            return ChartConfig(
                type=ChartType.BAR, data=[], xKey="hour", yKeys=["trades"]
            )

        # Pivot to get high/low side by side
        pivot = df_hourly.pivot_table(
            index="hour_utc", columns="market_type", values="n_trades", fill_value=0
        ).reset_index()

        chart_data = []
        for _, row in pivot.iterrows():
            entry = {"hour": f"{int(row['hour_utc']):02d}:00"}
            if "High" in pivot.columns:
                entry["High Temp"] = int(row.get("High", 0))
            if "Low" in pivot.columns:
                entry["Low Temp"] = int(row.get("Low", 0))
            chart_data.append(entry)

        y_keys = [k for k in ["High Temp", "Low Temp"] if k in chart_data[0]]

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="hour",
            yKeys=y_keys,
            title="Weather Market Trades by Hour (UTC)",
            xLabel="Hour of Day (UTC)",
            yLabel="Number of Trades",
            colors={"High Temp": "#d62728", "Low Temp": "#1f77b4"},
        )
