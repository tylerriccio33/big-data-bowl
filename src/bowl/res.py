"""Receiver success score ("res") from regular-season play-by-play.

Two components, each a percentile among drafted WRs:

1. epa_per_play - EPA on the receiver's targets per route run (untargeted routes count as 0)
2. target_share - receiver targets / team pass attempts in the games they played

res = 0.5 * epa_per_play + 0.5 * target_share, on a 0-100 scale.
"""

from pathlib import Path

import polars as pl

P = Path("data/parquet")
MIN_ROUTES_BOTTOM = 150


def shrink(value: pl.Expr, n: pl.Expr, k: float) -> pl.Expr:
    """Empirical-Bayes style shrinkage of a per-player mean toward the population mean."""
    prior = value.drop_nans().mean()
    return ((n * value.fill_null(prior) + k * prior) / (n + k)).alias(value.meta.output_name())


def pct(col: str) -> pl.Expr:
    return (pl.col(col).rank() / pl.col(col).count() * 100).alias(f"{col}_pct")


def load_plays() -> pl.LazyFrame:
    reg = pl.scan_parquet(P / "games.parquet").filter(pl.col("season_type") == "REG")
    return (
        pl.scan_parquet(P / "player_play.parquet")
        .join(reg.select("game_id"), on="game_id")
        .with_columns(target=pl.col("target").fill_null(False))
    )


def receiving(plays: pl.LazyFrame) -> pl.DataFrame:
    wrs = pl.scan_parquet(P / "players.parquet").filter(pl.col("nfl_position") == "WR")
    # The data covers only some receivers per team, so team pass attempts stand in for team targets.
    team_targets = (
        plays.filter(
            (pl.col("team_abbr") == pl.col("possession_team"))
            & pl.col("pass_result").is_in(["C", "I", "IN"])
        )
        .group_by("game_id", "team_abbr")
        .agg(team_targets=pl.col("play_id").n_unique())
    )
    routes = (
        plays.join(wrs.select("nfl_id"), on="nfl_id")
        .filter(pl.col("route_ran").is_not_null())
        .with_columns(
            rec_epa=pl.when(pl.col("target"))
            .then(pl.col("expected_points_added").fill_null(0))
            .otherwise(0),
            rec_yards=pl.col("rec_yards").fill_null(0),
        )
    )
    games = (
        routes.select("nfl_id", "game_id", "team_abbr")
        .unique()
        .join(team_targets, on=["game_id", "team_abbr"], how="left")
        .group_by("nfl_id")
        .agg(team_targets=pl.col("team_targets").fill_null(0).sum())
    )
    return (
        routes.group_by("nfl_id")
        .agg(
            routes=pl.len(),
            targets=pl.col("target").sum(),
            yards=pl.col("rec_yards").sum(),
            epa_per_play=pl.col("rec_epa").mean(),
        )
        .join(games, on="nfl_id", how="left")
        .with_columns(target_share=pl.col("targets") / pl.col("team_targets"))
        .with_columns(
            shrink(pl.col("epa_per_play"), pl.col("routes"), 200),
            shrink(pl.col("target_share"), pl.col("team_targets"), 150),
        )
        .collect()
    )


def build() -> pl.DataFrame:
    players = pl.read_parquet(P / "players.parquet").filter(pl.col("nfl_position") == "WR")
    df = players.select("nfl_id", "display_name", "draft_year", "draft_overall_pick").join(
        receiving(load_plays()), on="nfl_id", how="left"
    )
    # Players with no regular-season routes get the floor on every play-based measure.
    df = df.with_columns(pl.col("routes", "targets", "yards").fill_null(0))
    return (
        df.with_columns(pct("epa_per_play"), pct("target_share"))
        .with_columns(pl.col("epa_per_play_pct", "target_share_pct").fill_null(0))
        .with_columns(
            res=0.5 * pl.col("epa_per_play_pct") + 0.5 * pl.col("target_share_pct"),
        )
        .sort("res", descending=True)
    )


def main() -> None:
    df = build()
    df.write_parquet(P / "wr_res.parquet")
    cols = ["display_name", "draft_year", "draft_overall_pick", "routes", "yards"]
    cols += ["epa_per_play", "target_share", "epa_per_play_pct", "target_share_pct", "res"]
    with pl.Config(tbl_rows=20, tbl_cols=-1, float_precision=3, tbl_width_chars=200):
        print("TOP 10\n", df.head(10).select(cols))
        bottom = df.filter(pl.col("routes") >= MIN_ROUTES_BOTTOM).tail(10)
        print(f"BOTTOM 10 (>= {MIN_ROUTES_BOTTOM} routes)\n", bottom.select(cols))


if __name__ == "__main__":
    main()
