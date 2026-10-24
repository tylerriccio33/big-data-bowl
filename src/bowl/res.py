"""Receiver success score ("res") from regular-season play-by-play.

Two components, each a percentile among drafted WRs:

1. epa_per_play - EPA on the receiver's targets per route run (untargeted routes count as 0)
2. target_share - receiver targets / team targets in the games they ran a route

res = 0.5 * epa_per_play + 0.5 * target_share, on a 0-100 scale.
"""

from pathlib import Path

import polars as pl

P = Path("data/parquet")
MIN_ROUTES_BOTTOM = 150


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
    # Team targets: aimed passes on any play with a cohort player on either side of the ball,
    # which covers nearly every team pass. Throwaways and spikes name no receiver.
    team_targets = (
        plays.filter(
            pl.col("pass_result").is_in(["C", "I", "IN"])
            & pl.col("play_description").str.contains(" pass .* to ")
        )
        .group_by("game_id", team_abbr="possession_team")
        .agg(team_targets=pl.col("play_id").n_unique())
    )
    routes = (
        plays.join(wrs.select("nfl_id"), on="nfl_id")
        .filter(pl.col("route_ran").is_not_null())
        .with_columns(
            # Play EPA belongs to the receiver only when targeted; an untargeted route adds nothing.
            rec_epa=pl.when(pl.col("target")).then(pl.col("expected_points_added")).otherwise(0.0),
            rec_yards=pl.col("rec_yards").fill_null(0),
        )
    )
    games = (
        routes.select("nfl_id", "game_id", "team_abbr")
        .unique()
        .join(team_targets, on=["game_id", "team_abbr"], how="left")
        .group_by("nfl_id")
        .agg(team_targets=pl.col("team_targets").sum())
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
        .collect()
    )


def build() -> pl.DataFrame:
    players = pl.read_parquet(P / "players.parquet").filter(pl.col("nfl_position") == "WR")
    df = players.select("nfl_id", "display_name", "draft_year", "draft_overall_pick").join(
        receiving(load_plays()), on="nfl_id", how="left"
    )
    # Players with no regular-season routes have no res rather than a defaulted one.
    return (
        df.with_columns(pct("epa_per_play"), pct("target_share"))
        .with_columns(
            res=0.5 * pl.col("epa_per_play_pct") + 0.5 * pl.col("target_share_pct"),
        )
        .sort("res", descending=True, nulls_last=True)
    )


def main() -> None:
    df = build()
    df.write_parquet(P / "wr_res.parquet")
    cols = ["display_name", "draft_year", "draft_overall_pick", "routes", "yards"]
    cols += ["epa_per_play", "target_share", "epa_per_play_pct", "target_share_pct", "res"]
    with pl.Config(tbl_rows=20, tbl_cols=-1, float_precision=3, tbl_width_chars=200):
        print("TOP 10\n", df.head(10).select(cols))
        bottom = df.filter(pl.col("routes") >= MIN_ROUTES_BOTTOM).tail(10)
        print("no regular-season routes (res is null):", df["res"].null_count())
        print(f"BOTTOM 10 (>= {MIN_ROUTES_BOTTOM} routes)\n", bottom.select(cols))


if __name__ == "__main__":
    main()
