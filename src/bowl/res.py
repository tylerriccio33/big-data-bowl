"""Receiver success score ("res") from regular-season play-by-play.

Four skill pillars plus production and usage/trust, each a percentile among drafted WRs:

1. man        - beat man: separation over expected and yards per route vs man (no screens)
2. yac        - ball in hands: yards after catch over expected per reception
3. zone       - beat zone: target rate and yards per route vs zone coverage
4. hands      - catch rate over expected given air yards and separation
5. production - EPA per target
6. usage      - targets per route, games-active rate, start rate

Receivers need to be elite at one thing, so skill = 0.5 * best pillar + 0.5 * mean pillar.
res = 0.45 * skill + 0.35 * production + 0.2 * usage, on a 0-100 scale.
"""

from pathlib import Path

import polars as pl

P = Path("data/parquet")
REG_GAMES_PER_SEASON = 17
LAST_SEASON = 2025
PILLARS = ["man", "yac", "zone", "hands"]
MIN_ROUTES_BOTTOM = 150


def shrink(value: pl.Expr, n: pl.Expr, k: float) -> pl.Expr:
    """Empirical-Bayes style shrinkage of a per-player mean toward the population mean."""
    prior = value.drop_nans().mean()
    return ((n * value.fill_null(prior) + k * prior) / (n + k)).alias(value.meta.output_name())


def pct(col: str) -> pl.Expr:
    return (pl.col(col).rank() / pl.col(col).count() * 100).alias(f"{col}_pct")


def load_routes() -> pl.DataFrame:
    wrs = pl.scan_parquet(P / "players.parquet").filter(pl.col("nfl_position") == "WR")
    reg = pl.scan_parquet(P / "games.parquet").filter(pl.col("season_type") == "REG")
    return (
        pl.scan_parquet(P / "player_play.parquet")
        .join(wrs.select("nfl_id"), on="nfl_id")
        .join(reg.select("game_id", "season"), on="game_id")
        .filter(pl.col("route_ran").is_not_null())
        .with_columns(
            target=pl.col("target").fill_null(False),
            catch=pl.col("target").fill_null(False) & (pl.col("pass_result") == "C"),
            # Screens say little about beating a man defender, so they are left out of man.
            man=(pl.col("team_coverage_man_zone") == "MAN_COVERAGE")
            & (pl.col("route_ran") != "SCREEN"),
            zone=pl.col("team_coverage_man_zone") == "ZONE_COVERAGE",
            rec_yards=pl.col("rec_yards").fill_null(0),
            air_bucket=(pl.col("pass_length").clip(-5, 25) // 5),
            sep_bucket=pl.col("separation_at_pass_forward").clip(0, 6).floor(),
        )
        # Expected separation for the route and coverage; expected catch for depth and separation.
        .with_columns(
            sep_oe=pl.col("separation_at_pass_forward")
            - pl.col("separation_at_pass_forward").mean().over("route_ran", "man"),
        )
        .with_columns(
            catch_oe=pl.when(pl.col("target")).then(
                pl.col("catch").cast(pl.Float64)
                - pl.col("catch").filter(pl.col("target")).mean().over("air_bucket", "sep_bucket")
            ),
            yac_oe=pl.when(pl.col("catch")).then(
                pl.col("yards_after_catch") - pl.col("expected_yards_after_catch")
            ),
        )
        .collect()
    )


def pillars(routes: pl.DataFrame) -> pl.DataFrame:
    by_player = routes.group_by("nfl_id").agg(
        routes=pl.len(),
        games=pl.col("game_id").n_unique(),
        targets=pl.col("target").sum(),
        catches=pl.col("catch").sum(),
        yards=pl.col("rec_yards").sum(),
        man_n=pl.col("man").sum(),
        man_sep_oe=pl.col("sep_oe").filter(pl.col("man")).mean(),
        man_yprr=pl.col("rec_yards").filter(pl.col("man")).mean(),
        zone_n=pl.col("zone").sum(),
        zone_tprr=pl.col("target").filter(pl.col("zone")).mean(),
        zone_yprr=pl.col("rec_yards").filter(pl.col("zone")).mean(),
        yac_oe=pl.col("yac_oe").mean(),
        catch_oe=pl.col("catch_oe").mean(),
        epa_per_target=pl.col("expected_points_added").filter(pl.col("target")).mean(),
        tprr=pl.col("target").mean(),
    )
    return (
        by_player.with_columns(
            shrink(pl.col("man_sep_oe"), pl.col("man_n"), 120),
            shrink(pl.col("man_yprr"), pl.col("man_n"), 120),
            shrink(pl.col("zone_tprr"), pl.col("zone_n"), 200),
            shrink(pl.col("zone_yprr"), pl.col("zone_n"), 200),
            shrink(pl.col("yac_oe"), pl.col("catches"), 40),
            shrink(pl.col("catch_oe"), pl.col("targets"), 50),
            shrink(pl.col("epa_per_target"), pl.col("targets"), 50),
            shrink(pl.col("tprr"), pl.col("routes"), 200),
        )
        .with_columns(pct(c) for c in ["man_sep_oe", "man_yprr", "zone_tprr", "zone_yprr"])
        .with_columns(
            man=(pl.col("man_sep_oe_pct") + pl.col("man_yprr_pct")) / 2,
            zone=(pl.col("zone_tprr_pct") + pl.col("zone_yprr_pct")) / 2,
            yac=pct("yac_oe").name.keep(),
            hands=pct("catch_oe").name.keep(),
            production=pct("epa_per_target").name.keep(),
        )
    )


def usage() -> pl.DataFrame:
    players = pl.read_parquet(P / "players.parquet").filter(pl.col("nfl_position") == "WR")
    career = pl.read_parquet(P / "player_career_successes.parquet")
    avail = (LAST_SEASON - pl.col("draft_year") + 1) * REG_GAMES_PER_SEASON
    return players.join(career, on="nfl_id").select(
        "nfl_id",
        "display_name",
        "draft_year",
        "draft_overall_pick",
        active_rate=pl.col("career_games_active") / avail,
        start_rate=pl.col("career_games_started") / avail,
        accolades=pl.col("ap_all_pro_1st_team") * 2
        + pl.col("ap_all_pro_2nd_team")
        + pl.col("pro_bowl_original_ballot"),
    )


def build() -> pl.DataFrame:
    skill = pillars(load_routes())
    df = usage().join(skill, on="nfl_id", how="left")
    # Players with no regular-season routes get the floor on every play-based measure.
    df = df.with_columns(
        pl.col(*PILLARS, "production").fill_null(0),
        tprr=pl.col("tprr").fill_null(0),
        routes=pl.col("routes").fill_null(0),
        yards=pl.col("yards").fill_null(0),
    )
    return (
        df.with_columns(pct(c) for c in ["tprr", "active_rate", "start_rate"])
        .with_columns(
            usage=pl.mean_horizontal("tprr_pct", "active_rate_pct", "start_rate_pct"),
            skill=0.5 * pl.max_horizontal(*PILLARS) + 0.5 * pl.mean_horizontal(*PILLARS),
        )
        .with_columns(
            res=(
                0.45 * pl.col("skill")
                + 0.35 * pl.col("production")
                + 0.2 * pl.col("usage")
                + 3 * pl.col("accolades")
            ).clip(0, 100)
        )
        .sort("res", descending=True)
    )


def main() -> None:
    df = build()
    df.write_parquet(P / "wr_res.parquet")
    cols = ["display_name", "draft_year", "draft_overall_pick", "routes", "yards"]
    cols += [*PILLARS, "production", "usage", "res"]
    with pl.Config(tbl_rows=20, tbl_cols=-1, float_precision=0, tbl_width_chars=200):
        print("TOP 10\n", df.head(10).select(cols))
        bottom = df.filter(pl.col("routes") >= MIN_ROUTES_BOTTOM).tail(10)
        print(f"BOTTOM 10 (>= {MIN_ROUTES_BOTTOM} routes)\n", bottom.select(cols))


if __name__ == "__main__":
    main()
