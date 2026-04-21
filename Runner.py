#!/bin/python3
import pandas as pd
import matplotlib.pyplot as plt
import Model
import Prediction

class Reader:
    def get_dataframe(self, file: str) -> pd.DataFrame:
        raise NotImplementedError

class ReaderFootballDataCoUK(Reader):
    def __init__(self):
        self.headers_to_remove = [
            "Country",
            "League",
            "Season",
            "Time",
        ]

    def _get_raw_dataframe(self, file: str) -> pd.DataFrame:
        return pd.read_csv(file, sep=",")

    def _clean_and_order_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("Date", ascending=True)
        df = df.drop(columns=self.headers_to_remove)
        return df

    def _keep_only_best_bets(self, df:pd.DataFrame)->pd.DataFrame:
        home_cols = [col for col in df.columns if "CH" in col]
        tie_cols = [col for col in df.columns if "CD" in col]
        away_cols = [col for col in df.columns if "CA" in col]
        best_bets = pd.DataFrame({
            "home_decimal": df[home_cols].max(axis=1),
            "tie_decimal": df[tie_cols].max(axis=1),
            "away_decimal": df[away_cols].max(axis=1)
        })
        df = df.drop(columns=home_cols + tie_cols + away_cols)
        return pd.concat([df, best_bets], axis=1)

    # ett gemensamt format som alla ska ha - detta måste fixas
    def _convert_to_format(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def get_dataframe(self, file) -> pd.DataFrame:
        df = self._get_raw_dataframe(file)
        df = self._clean_and_order_data(df)
        df = self._keep_only_best_bets(df)
        df = self._convert_to_format(df)
        return df


class StateInitializer:
    def __init__(self, reader: Reader, default_skill:Model.GeneralSkill|Model.AttackDefenseSkill):
        self.reader = reader
        self.default_skill = default_skill

    def _get_unique_teams_from_df(self,
        df: pd.DataFrame,
    ):
        all_teams = pd.concat([df["Home"], df["Away"]]).unique()
        return all_teams

    def initialize_modelstate(self, file: str):
        df = self.reader.get_dataframe(file)
        all_teams = self._get_unique_teams_from_df(df)
        state = Model.ModelState(teams={team: self.default_skill for team in all_teams})
        return df, state


# sloppy test code
def test():
    print("viktigt kolla så inte alla team får samma referens till skill.. tror ej det då immutable?")
    initial_team_dist = Model.NormalDistribution(Model.DEFAULT_START_MU, Model.DEFAULT_START_VAR)
    initial_home_advantage_dist = Model.NormalDistribution(0, 10)
    default_dist = Model.NormalDistribution(Model.DEFAULT_START_MU, Model.DEFAULT_START_VAR)
    df, state = StateInitializer(ReaderFootballDataCoUK(), Model.AttackDefenseSkill(default_dist, default_dist)).initialize_modelstate("./datasets/SWE.csv")
    # df, state = StateInitializer(ReaderFootballDataCoUK(), Model.GeneralSkill(default_dist)).initialize_modelstate("./datasets/SWE.csv")
    team_count = {k:0 for k in state.teams.keys()}
    state.home_advantage = initial_home_advantage_dist
    # print(df, state)
    # print(state.teams.keys())
    # print(state.teams.get("AIK"))
    # print(state.teams.get("Orebro"))

    model = Model.TrueskillAdvancedAttackDefense(Model.DEFAULT_VAR_T, Model.DEFAULT_VAR_T)
    adapter = Model.TrueskillAdvancedAttackDefenseAdapter(model)
    # model = Model.TrueskillAdvanced(Model.DEFAULT_VAR_T, 1)
    # adapter = Model.TrueskillAdvancedAdapter(model)
    engine = Model.GameEngine(adapter, state)
    state_history = []
    dates = []
    df_train = df.iloc[:500]
    df_test = df.iloc[500:]
    print("Training....")
    for inx, row in df_train.iterrows():
        home = str(row["Home"])
        away = str(row["Away"])
        team_count[home] += 1
        team_count[away] += 1
        game = Model.Game(home, away, Model.Score(int(row["HG"]), int(row["AG"])))
        result = engine.update(game)
        state_history.append(engine.state)
        dates.append(row["Date"])
    print("Training complete....")
    print(engine.state)

    print("Training and predicting....")
    strategy = Prediction.ValueBetStrategy(4.0, Prediction.KellyBetScaler())
    # total, correct = 0,0
    budget = 100
    b_lst = [budget]
    # TODO: De måste ha kört ett antal gånger först
    for inx, row in df_test.iterrows():
        home = str(row["Home"])
        away = str(row["Away"])
        game = Model.Game(home, away)
        pred = engine.predict(game)
        market_odds = Prediction.MarketOdds.from_floats(row["home_decimal"], row["away_decimal"], row["tie_decimal"])
        if team_count[home] > 10 and team_count[away] > 10:
            bet_lst = strategy.decide(pred, market_odds)
        else:
            print(f"skipping bet on {game}")
            bet_lst = []
        game = Model.Game(home, away, Model.Score(int(row["HG"]), int(row["AG"])))
        engine.update(game)
        team_count[home] += 1
        team_count[away] += 1
        for bet in bet_lst:
            bet_delta = Prediction.evaluate_bet(bet, game.score.outcome)
            budget = budget + bet_delta
            # print(f"({bet_delta}) Placed bet on {bet.side} and outcome was {game.score.outcome}")
            b_lst.append(budget)
    # print(f"{correct}/{total} = {correct/total}")

    print(engine.state)
    plt.figure(1)
    plt.plot(b_lst)
    # teams = [s.teams for s in state_history]
    # for team in state.teams.keys():
    #     plt.plot(dates, [t[team].attack.mean for t in teams], label=team)
    # plt.legend()
    plt.show()

test()
