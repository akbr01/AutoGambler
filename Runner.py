#!/bin/python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
import Model
import Prediction

class Reader:
    def get_dataframe(self, file: str) -> pd.DataFrame:
        raise NotImplementedError


class ReaderGeneral(Reader):
    def __init__(self, headers_to_remove: list[str], home_bet_re:str, tie_bet_re:str, away_bet_re:str, date:str, home:str, away:str, hg:str, ag:str, date_format:str):
        self.headers_to_remove = headers_to_remove
        self.home_bet_re = home_bet_re
        self.tie_bet_re = tie_bet_re
        self.away_bet_re = away_bet_re
        self.date = date
        self.home = home
        self.away = away
        self.hg = hg
        self.ag = ag
        self.date_format = date_format

    def _get_raw_dataframe(self, file: str) -> pd.DataFrame:
        return pd.read_csv(file, sep=",")

    def _clean_and_order_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("Date", ascending=True)
        df = df.drop(columns=self.headers_to_remove)
        return df

    def _keep_only_best_bets(self, df:pd.DataFrame)->pd.DataFrame:
        home_cols = [col for col in df.columns if re.search(self.home_bet_re, col, )]
        tie_cols = [col for col in df.columns if re.search(self.tie_bet_re, col, )]
        away_cols = [col for col in df.columns if re.search(self.away_bet_re, col, )]
        best_bets = pd.DataFrame({
            "home_decimal": df[home_cols].max(axis=1),
            "tie_decimal": df[tie_cols].max(axis=1),
            "away_decimal": df[away_cols].max(axis=1)
        })
        df = df.drop(columns=home_cols + tie_cols + away_cols)
        return pd.concat([df, best_bets], axis=1)

    def _convert_to_format(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={self.home: "Home", self.away: "Away", self.hg: "HG", self.ag: "AG", self.date: "Date"})
        df = df.astype({"Home": str, "Away": str, "HG": int, "AG": int}) 
        df["Date"] = pd.to_datetime(df["Date"], format=self.date_format)
        return df

    def get_dataframe(self, file) -> pd.DataFrame:
        df = self._get_raw_dataframe(file)
        df = self._convert_to_format(df)
        df = self._clean_and_order_data(df)
        df = self._keep_only_best_bets(df)
        return df


class StateInitializer:
    def __init__(self, default_skill: Model.GeneralSkill | Model.AttackDefenseSkill):
        self.default_skill = default_skill

    def initialize_modelstate(self, df: pd.DataFrame) -> Model.ModelState:
        all_teams = pd.concat([df["Home"], df["Away"]]).unique()
        return Model.ModelState(teams={team: self.default_skill for team in all_teams})


def merge_datasets(*pairs: tuple[Reader, str]) -> pd.DataFrame:
    dfs = [reader.get_dataframe(path) for reader, path in pairs]
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged


def apply_skill_decay(state: Model.ModelState, days_elapsed: int, decay_rate: float = 0.5):
    for team_name in list(state.teams.keys()):
        skill = state.teams[team_name]
        if isinstance(skill, Model.GeneralSkill):
            new_nd = Model.NormalDistribution(skill.skill.mean, skill.skill.variance + decay_rate * days_elapsed)
            state.teams[team_name] = Model.GeneralSkill(skill=new_nd)
        elif isinstance(skill, Model.AttackDefenseSkill):
            new_atk = Model.NormalDistribution(skill.attack.mean, skill.attack.variance + decay_rate * days_elapsed)
            new_def = Model.NormalDistribution(skill.defense.mean, skill.defense.variance + decay_rate * days_elapsed)
            state.teams[team_name] = Model.AttackDefenseSkill(attack=new_atk, defense=new_def)


# sloppy test code
def test():
    initial_team_dist = Model.NormalDistribution(Model.DEFAULT_START_MU, Model.DEFAULT_START_VAR)
    initial_home_advantage_dist = Model.NormalDistribution(0.16, 10)
    default_dist = Model.NormalDistribution(Model.DEFAULT_START_MU, Model.DEFAULT_START_VAR)

    # Datasets: list of (reader, filepath) pairs
    datasets = [
        (ReaderGeneral(["leagueID", "season"], r"C?H$", r"^(?!.*ID$).*C?D$", r"C?A$", "date", "homeTeamID", "awayTeamID", "homeGoals", "awayGoals", "%Y-%m-%d %H:%M:%S"), "./datasets/games.csv"),
        # (ReaderGeneral(["League", "Season"], r"C?H$", r"^(?!.*ID$).*C?D$", r"C?A$", "Date", "Home", "Away", "HG", "AG", "%d/%m/%Y"), "./datasets/SWE.csv"),
    ]

    train_fraction = 0.8
    decay_rate = 0.1
    min_observed_games_to_play = 20
    bet_threshold = 5
    kelly_multiplier = 0.2

    # Merge and sort all datasets by date
    df = merge_datasets(*datasets)
    date_cutoff = df["Date"].quantile(train_fraction)
    print(f"Training cutoff: {date_cutoff}")

    state = StateInitializer(Model.AttackDefenseSkill(default_dist, default_dist)).initialize_modelstate(df)
    model = Model.TrueskillAdvancedAttackDefense(Model.DEFAULT_VAR_T, Model.DEFAULT_VAR_T)
    adapter = Model.TrueskillAdvancedAttackDefenseAdapter(model)
    state.home_advantage = initial_home_advantage_dist
    engine = Model.GameEngine(adapter, state)

    strategy = Prediction.ValueBetStrategy(bet_threshold, Prediction.KellyBetScaler(kelly_multiplier))

    team_count = {k: 0 for k in state.teams.keys()}
    prev_date = None
    budget = 100
    budget_history = [budget]
    date_history = [df["Date"].iloc[0]]

    print("Running date-driven loop...")
    for date, group in df.groupby("Date", sort=True):
        days_elapsed = (date - prev_date).days if prev_date is not None else 0
        prev_date = date

        if days_elapsed > 0 and decay_rate > 0:
            apply_skill_decay(engine.state, days_elapsed, decay_rate)

        is_test = date >= date_cutoff

        for _, row in group.iterrows():
            home = str(row["Home"])
            away = str(row["Away"])
            team_count[home] += 1
            team_count[away] += 1

            if is_test:
                game_no_score = Model.Game(home, away)
                pred = engine.predict(game_no_score)
                market_odds = Prediction.MarketOdds.from_floats(row["home_decimal"], row["away_decimal"], row["tie_decimal"])

                if team_count[home] > min_observed_games_to_play and team_count[away] > min_observed_games_to_play:
                    bet_lst = strategy.decide(pred, market_odds)
                else:
                    bet_lst = []

            game = Model.Game(home, away, Model.Score(int(row["HG"]), int(row["AG"])))
            engine.update(game)

            if is_test:
                for bet in bet_lst:
                    bet_delta = Prediction.evaluate_bet(bet, game.score.outcome)
                    budget += bet_delta
                    budget_history.append(budget)
                    date_history.append(date)

    print(f"Final budget: {budget:.2f}")
    print(f"Bets placed: {len(budget_history) - 1}")

    fig, ax = plt.subplots()
    ax.plot(date_history, budget_history)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    plt.show()


def search_params():
    import Optimization as Opt

    initial_home_advantage_dist = Model.NormalDistribution(0.16, 10)
    default_dist = Model.NormalDistribution(Model.DEFAULT_START_MU, Model.DEFAULT_START_VAR)

    datasets = [
        (ReaderGeneral(["leagueID", "season"], r"C?H$", r"^(?!.*ID$).*C?D$", r"C?A$", "date", "homeTeamID", "awayTeamID", "homeGoals", "awayGoals", "%Y-%m-%d %H:%M:%S"), "./datasets/games.csv"),
    ]
    df = merge_datasets(*datasets)

    param_space = {
        "var_t": Opt.UniformFloat(1, 100),
        "var_y": Opt.UniformFloat(1, 100),
        # "train_fraction": Opt.UniformFloat(0.5, 0.95),
        "train_fraction": 0.7,
        "decay_rate": Opt.UniformFloat(0, 5),
        "bet_threshold": Opt.UniformFloat(0, 1),
        "kelly_multiplier": Opt.UniformFloat(0, 1),
        "min_games": Opt.UniformInt(1, 50),
    }


    def eval_fn(params: dict) -> float:
        state = StateInitializer(Model.AttackDefenseSkill(default_dist, default_dist)).initialize_modelstate(df)
        model = Model.TrueskillAdvancedAttackDefense(params["var_t"], params["var_y"])
        adapter = Model.TrueskillAdvancedAttackDefenseAdapter(model)
        state.home_advantage = initial_home_advantage_dist
        engine = Model.GameEngine(adapter, state)

        strategy = Prediction.ValueBetStrategy(params["bet_threshold"], Prediction.KellyBetScaler(params["kelly_multiplier"]))

        team_count = {k: 0 for k in state.teams.keys()}
        prev_date = None
        budget = 100
        date_cutoff = df["Date"].quantile(params["train_fraction"])

        for date, group in df.groupby("Date", sort=True):
            days_elapsed = (date - prev_date).days if prev_date is not None else 0
            prev_date = date
            if days_elapsed > 0 and params["decay_rate"] > 0:
                apply_skill_decay(engine.state, days_elapsed, params["decay_rate"])
            is_test = date >= date_cutoff
            for _, row in group.iterrows():
                home = str(row["Home"])
                away = str(row["Away"])
                team_count[home] += 1
                team_count[away] += 1
                if is_test:
                    game_no_score = Model.Game(home, away)
                    pred = engine.predict(game_no_score)
                    market_odds = Prediction.MarketOdds.from_floats(row["home_decimal"], row["away_decimal"], row["tie_decimal"])
                    if team_count[home] > params["min_games"] and team_count[away] > params["min_games"]:
                        bet_lst = strategy.decide(pred, market_odds)
                    else:
                        bet_lst = []
                game = Model.Game(home, away, Model.Score(int(row["HG"]), int(row["AG"])))
                engine.update(game)
                if is_test:
                    for bet in bet_lst:
                        bet_delta = Prediction.evaluate_bet(bet, game.score.outcome)
                        budget += bet_delta
        return budget

    searcher = Opt.RandomSearch(
        param_space=param_space,
        eval_fn=eval_fn,
        n_trials=30,
        maximize=True,
        seed=42,
    )
    searcher.search(plot=True)


# test()

if __name__ == "__main__":
    search_params()
