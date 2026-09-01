"""Analyze weather market trade activity around hourly METAR observation releases.

This is the core alpha-finding analysis. The hypothesis is:

1. Weather stations report METAR observations roughly every hour (typically at :53)
2. There is a processing/propagation lag before this data reaches traders
3. Traders who see the new observation first can trade on the updated temperature
   before the market fully adjusts
4. This should manifest as:
   - Volume spikes shortly after :53 each hour
   - Directional price movement (yes_price changes) clustered after observations
   - Better excess returns for trades placed in the minutes *immediately after*
     an observation vs trades placed at other times

The analysis uses minute-level granularity to identify the exact lag window.

Key outputs:
- Trade volume by minute-within-hour (are there spikes after :53?)
- Price movement (absolute delta in yes_price) by minute-within-hour
- Excess returns by minute-within-hour (is there alpha in the first few minutes
  after an observation release?)
- Comparison of "informed" window (e.g. :54-:59) vs "stale" window (e.g. :20-:50)
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


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


class WeatherTradeTimingAnalysis(Analysis):
    """Analyze trade timing around hourly weather observation releases.

    Looks for the signature of informed trading: volume spikes and
    excess returns clustered in the minutes after METAR observations
    (typically released at :53 past each hour).
    """

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="weather_trade_timing",
            description="Trade timing around hourly METAR releases in weather markets",
        )
        base_dir = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()
        weather_filter = _weather_filter_sql()

        # --- Core dataset: minute-level trade data for weather markets ---
        with self.progress("Loading weather trade data at minute granularity"):
            df_minute = con.execute(
                f"""
                WITH weather_markets AS (
                    SELECT ticker, event_ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE {weather_filter}
                      AND status = 'finalized'
                      AND result IN ('yes', 'no')
                ),
                trade_data AS (
                    SELECT
                        t.created_time,
                        t.ticker,
                        t.yes_price,
                        t.no_price,
                        t.taker_side,
                        t.count AS contracts,
                        m.event_ticker,
                        m.result,
                        EXTRACT(MINUTE FROM t.created_time) AS minute_of_hour,
                        EXTRACT(HOUR FROM t.created_time) AS hour_utc,
                        -- Taker's price and outcome
                        CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                             ELSE t.no_price END AS taker_price,
                        CASE WHEN t.taker_side = m.result THEN 1.0
                             ELSE 0.0 END AS taker_won,
                        -- Volume
                        t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                                   ELSE t.no_price END) / 100.0 AS volume_usd
                    FROM '{self.trades_dir}/*.parquet' t
                    INNER JOIN weather_markets m ON t.ticker = m.ticker
                )
                SELECT
                    minute_of_hour,
                    hour_utc,
                    COUNT(*) AS n_trades,
                    SUM(contracts) AS total_contracts,
                    SUM(volume_usd) AS total_volume_usd,
                    AVG(taker_price / 100.0) AS avg_implied_prob,
                    AVG(taker_won) AS win_rate,
                    AVG(taker_won - taker_price / 100.0) AS excess_return,
                    VAR_SAMP(taker_won - taker_price / 100.0) AS var_excess,
                    AVG(taker_price) AS avg_price_cents
                FROM trade_data
                GROUP BY minute_of_hour, hour_utc
                ORDER BY minute_of_hour, hour_utc
                """
            ).df()

        # --- Aggregate by minute-of-hour only (across all hours) ---
        with self.progress("Computing minute-of-hour aggregates"):
            df_by_minute = con.execute(
                f"""
                WITH weather_markets AS (
                    SELECT ticker, event_ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE {weather_filter}
                      AND status = 'finalized'
                      AND result IN ('yes', 'no')
                ),
                trade_data AS (
                    SELECT
                        EXTRACT(MINUTE FROM t.created_time) AS minute_of_hour,
                        t.count AS contracts,
                        CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                             ELSE t.no_price END AS taker_price,
                        CASE WHEN t.taker_side = m.result THEN 1.0
                             ELSE 0.0 END AS taker_won,
                        t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                                   ELSE t.no_price END) / 100.0 AS volume_usd
                    FROM '{self.trades_dir}/*.parquet' t
                    INNER JOIN weather_markets m ON t.ticker = m.ticker
                )
                SELECT
                    minute_of_hour,
                    COUNT(*) AS n_trades,
                    SUM(contracts) AS total_contracts,
                    SUM(volume_usd) AS total_volume_usd,
                    AVG(taker_price / 100.0) AS avg_implied_prob,
                    AVG(taker_won) AS win_rate,
                    AVG(taker_won - taker_price / 100.0) AS excess_return,
                    VAR_SAMP(taker_won - taker_price / 100.0) AS var_excess
                FROM trade_data
                GROUP BY minute_of_hour
                ORDER BY minute_of_hour
                """
            ).df()

        # Standard error and confidence intervals
        if len(df_by_minute) > 0:
            df_by_minute["se"] = np.sqrt(
                df_by_minute["var_excess"] / df_by_minute["n_trades"]
            )
            df_by_minute["ci_lower"] = (
                df_by_minute["excess_return"] - 1.96 * df_by_minute["se"]
            )
            df_by_minute["ci_upper"] = (
                df_by_minute["excess_return"] + 1.96 * df_by_minute["se"]
            )

        # --- Window comparison: "post-METAR" vs "pre-METAR" ---
        # METAR typically at :53. Test various lag windows.
        with self.progress("Computing window comparison stats"):
            df_windows = con.execute(
                f"""
                WITH weather_markets AS (
                    SELECT ticker, event_ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE {weather_filter}
                      AND status = 'finalized'
                      AND result IN ('yes', 'no')
                ),
                trade_data AS (
                    SELECT
                        EXTRACT(MINUTE FROM t.created_time) AS minute_of_hour,
                        t.count AS contracts,
                        CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                             ELSE t.no_price END AS taker_price,
                        CASE WHEN t.taker_side = m.result THEN 1.0
                             ELSE 0.0 END AS taker_won,
                        t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                                   ELSE t.no_price END) / 100.0 AS volume_usd
                    FROM '{self.trades_dir}/*.parquet' t
                    INNER JOIN weather_markets m ON t.ticker = m.ticker
                )
                SELECT
                    CASE
                        -- Minutes 53-59: the "METAR just dropped" window
                        WHEN minute_of_hour BETWEEN 53 AND 59 THEN 'A: :53-:59 (METAR release)'
                        -- Minutes 0-5: the "METAR +7 to +12 min" window (possible lag)
                        WHEN minute_of_hour BETWEEN 0 AND 5 THEN 'B: :00-:05 (post-METAR lag)'
                        -- Minutes 6-15: settling period
                        WHEN minute_of_hour BETWEEN 6 AND 15 THEN 'C: :06-:15 (settling)'
                        -- Minutes 16-52: "stale" period before next METAR
                        ELSE 'D: :16-:52 (stale/quiet)'
                    END AS time_window,
                    COUNT(*) AS n_trades,
                    SUM(contracts) AS total_contracts,
                    SUM(volume_usd) AS total_volume_usd,
                    AVG(taker_won) AS win_rate,
                    AVG(taker_price / 100.0) AS avg_implied_prob,
                    AVG(taker_won - taker_price / 100.0) AS excess_return,
                    VAR_SAMP(taker_won - taker_price / 100.0) AS var_excess
                FROM trade_data
                GROUP BY time_window
                ORDER BY time_window
                """
            ).df()

        if len(df_windows) > 0:
            df_windows["se"] = np.sqrt(
                df_windows["var_excess"] / df_windows["n_trades"]
            )
            df_windows["ci_lower"] = (
                df_windows["excess_return"] - 1.96 * df_windows["se"]
            )
            df_windows["ci_upper"] = (
                df_windows["excess_return"] + 1.96 * df_windows["se"]
            )
            # Volume per minute in each window
            window_sizes = {"A": 7, "B": 6, "C": 10, "D": 37}
            df_windows["minutes_in_window"] = (
                df_windows["time_window"].str[0].map(window_sizes)
            )
            df_windows["volume_per_minute"] = (
                df_windows["total_volume_usd"] / df_windows["minutes_in_window"]
            )
            df_windows["trades_per_minute"] = (
                df_windows["n_trades"] / df_windows["minutes_in_window"]
            )

        fig = self._create_figure(df_by_minute, df_windows, df_minute)
        chart = self._create_chart(df_by_minute)

        return AnalysisOutput(
            figure=fig,
            data=df_by_minute,
            chart=chart,
            metadata={
                "windows": df_windows.to_dict("records") if len(df_windows) > 0 else [],
            },
        )

    def _create_figure(
        self,
        df_by_minute: pd.DataFrame,
        df_windows: pd.DataFrame,
        df_minute: pd.DataFrame,
    ) -> plt.Figure:
        """Create a 2x2 figure showing the timing edge analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Weather Market Trade Timing Around METAR Releases",
            fontsize=16,
            fontweight="bold",
        )

        # --- Top-left: Trade volume by minute of hour ---
        ax = axes[0, 0]
        if len(df_by_minute) > 0:
            minutes = df_by_minute["minute_of_hour"].values
            trades = df_by_minute["n_trades"].values

            # Color bars: highlight the post-METAR window
            colors = []
            for m in minutes:
                if 53 <= m <= 59:
                    colors.append("#d62728")  # Red: METAR release window
                elif 0 <= m <= 5:
                    colors.append("#ff7f0e")  # Orange: post-METAR lag
                elif 6 <= m <= 15:
                    colors.append("#2ca02c")  # Green: settling
                else:
                    colors.append("#1f77b4")  # Blue: stale/quiet

            ax.bar(minutes, trades, color=colors, alpha=0.8, width=0.8)
            ax.axvline(x=53, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
            ax.annotate(
                "METAR\n:53",
                xy=(53, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 1),
                fontsize=9,
                color="red",
                ha="center",
                fontweight="bold",
            )
            ax.set_xlabel("Minute of Hour")
            ax.set_ylabel("Number of Trades")
            ax.set_title("Trade Volume by Minute of Hour")
            ax.set_xlim(-0.5, 59.5)
            ax.set_xticks(range(0, 60, 5))
            ax.grid(True, alpha=0.3, axis="y")

            # Add legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="#d62728", alpha=0.8, label=":53-:59 METAR release"),
                Patch(facecolor="#ff7f0e", alpha=0.8, label=":00-:05 post-METAR lag"),
                Patch(facecolor="#2ca02c", alpha=0.8, label=":06-:15 settling"),
                Patch(facecolor="#1f77b4", alpha=0.8, label=":16-:52 quiet"),
            ]
            ax.legend(handles=legend_elements, fontsize=7, loc="upper left")
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title("Trade Volume by Minute of Hour")

        # --- Top-right: Excess return by minute of hour ---
        ax = axes[0, 1]
        if len(df_by_minute) > 0:
            minutes = df_by_minute["minute_of_hour"].values
            excess = df_by_minute["excess_return"].values * 100  # pp
            ci_lo = df_by_minute["ci_lower"].values * 100
            ci_hi = df_by_minute["ci_upper"].values * 100

            ax.fill_between(minutes, ci_lo, ci_hi, alpha=0.2, color="#4C72B0")
            ax.plot(
                minutes,
                excess,
                color="#4C72B0",
                linewidth=1.5,
                marker=".",
                markersize=4,
            )
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axvline(x=53, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
            ax.annotate(
                "METAR\n:53",
                xy=(
                    53,
                    (
                        ax.get_ylim()[1] * 0.85
                        if ax.get_ylim()[1] != ax.get_ylim()[0]
                        else 1
                    ),
                ),
                fontsize=9,
                color="red",
                ha="center",
                fontweight="bold",
            )
            ax.set_xlabel("Minute of Hour")
            ax.set_ylabel("Excess Return (pp)")
            ax.set_title("Taker Excess Return by Minute of Hour")
            ax.set_xlim(-0.5, 59.5)
            ax.set_xticks(range(0, 60, 5))
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title("Taker Excess Return by Minute of Hour")

        # --- Bottom-left: Window comparison ---
        ax = axes[1, 0]
        if len(df_windows) > 0:
            short_labels = [
                w.split("(")[1].rstrip(")") for w in df_windows["time_window"]
            ]
            excess = df_windows["excess_return"].values * 100
            ci_lo = df_windows["ci_lower"].values * 100
            ci_hi = df_windows["ci_upper"].values * 100
            errs = np.array([excess - ci_lo, ci_hi - excess])

            window_colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
            bars = ax.bar(
                range(len(short_labels)),
                excess,
                color=window_colors[: len(short_labels)],
                alpha=0.8,
                yerr=errs,
                capsize=5,
                ecolor="gray",
            )
            ax.set_xticks(range(len(short_labels)))
            ax.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=9)
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_ylabel("Excess Return (pp)")
            ax.set_title("Taker Excess Return by Time Window")
            ax.grid(True, alpha=0.3, axis="y")

            # Annotate with trade counts
            for i, (_, row) in enumerate(df_windows.iterrows()):
                ax.annotate(
                    f"n={int(row['n_trades']):,}",
                    xy=(i, excess[i]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color="gray",
                )
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title("Excess Return by Window")

        # --- Bottom-right: Volume per minute by window ---
        ax = axes[1, 1]
        if len(df_windows) > 0 and "trades_per_minute" in df_windows.columns:
            short_labels = [
                w.split("(")[1].rstrip(")") for w in df_windows["time_window"]
            ]
            tpm = df_windows["trades_per_minute"].values

            window_colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
            ax.bar(
                range(len(short_labels)),
                tpm,
                color=window_colors[: len(short_labels)],
                alpha=0.8,
            )
            ax.set_xticks(range(len(short_labels)))
            ax.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=9)
            ax.set_ylabel("Trades per Minute")
            ax.set_title("Trade Intensity by Time Window")
            ax.grid(True, alpha=0.3, axis="y")

            # Show relative intensity
            baseline = tpm[-1] if len(tpm) > 0 and tpm[-1] > 0 else 1
            for i, val in enumerate(tpm):
                ratio = val / baseline
                ax.annotate(
                    f"{ratio:.1f}x",
                    xy=(i, val),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    fontweight="bold" if ratio > 1.5 else "normal",
                    color="#d62728" if ratio > 1.5 else "gray",
                )
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title("Trade Intensity by Window")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def _create_chart(self, df_by_minute: pd.DataFrame) -> ChartConfig:
        """Create chart config showing volume and excess return by minute."""
        if len(df_by_minute) == 0:
            return ChartConfig(
                type=ChartType.BAR, data=[], xKey="minute", yKeys=["trades"]
            )

        chart_data = [
            {
                "minute": f":{int(row['minute_of_hour']):02d}",
                "Trades": int(row["n_trades"]),
                "Excess Return (pp)": round(row["excess_return"] * 100, 3),
            }
            for _, row in df_by_minute.iterrows()
        ]

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="minute",
            yKeys=["Trades"],
            title="Weather Market Trades by Minute of Hour",
            xLabel="Minute of Hour",
            yLabel="Number of Trades",
        )
