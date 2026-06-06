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
        print(df)
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


# "gameID","leagueID","season","date","homeTeamID","awayTeamID","homeGoals","awayGoals","homeProbability","drawProbability","awayProbability","homeGoalsHalfTime","awayGoalsHalfTime","B365H","B365D","B365A","BWH","BWD","BWA","IWH","IWD","IWA","PSH","PSD","PSA","WHH","WHD","WHA","VCH","VCD","VCA","PSCH","PSCD","PSCA"

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

    # ett gemensamt format som alla ska ha - detta måste fixas
    # "gameID","leagueID","season","date","homeTeamID","awayTeamID","homeGoals","awayGoals","homeProbability","drawProbability","awayProbability","homeGoalsHalfTime","awayGoalsHalfTime","B365H","B365D","B365A","BWH","BWD","BWA","IWH","IWD","IWA","PSH","PSD","PSA","WHH","WHD","WHA","VCH","VCD","VCA","PSCH","PSCD","PSCA"

    # Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH,PSCD,PSCA,MaxCH,MaxCD,MaxCA,AvgCH,AvgCD,AvgCA,BFECH,BFECD,BFECA,B365CH,B365CD,B365CA
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
    initial_team_dist = Model.NormalDistribution(Model.DEFAULT_START_MU, Model.DEFAULT_START_VAR)
    initial_home_advantage_dist = Model.NormalDistribution(0, 10)
    default_dist = Model.NormalDistribution(Model.DEFAULT_START_MU, Model.DEFAULT_START_VAR)

    reader = ReaderGeneral(["leagueID", "season"], r"C?H$", r"^(?!.*ID$).*C?D$", r"C?A$", "date", "homeTeamID", "awayTeamID", "homeGoals", "awayGoals", "%Y-%m-%d %H:%M:%S")
    datafile = "./datasets/games.csv" # kaggle

    # reader = ReaderGeneral(["League", "Season"], r"C?H$", r"^(?!.*ID$).*C?D$", r"C?A$", "Date", "Home", "Away", "HG", "AG", "%d/%m/%Y")
    # datafile = "./datasets/SWE.csv" #  footballfatacouk

    df, state = StateInitializer(reader, Model.AttackDefenseSkill(default_dist, default_dist)).initialize_modelstate(datafile)

    print(df)

    # df, state = StateInitializer(ReaderFootballDataCoUK(), Model.GeneralSkill(default_dist)).initialize_modelstate("./datasets/SWE.csv")
    team_count = {k:0 for k in state.teams.keys()}
    print(team_count)
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
    print("Training complete....")
    print(engine.state)

    print("Training and predicting....")
    strategy = Prediction.ValueBetStrategy(0.05, Prediction.KellyBetScaler())
    # total, correct = 0,0
    budget = 100
    b_lst = [budget]
    dates = [df_test.iloc[0]["Date"]]
    for inx, row in df_test.iterrows():
        home = str(row["Home"])
        away = str(row["Away"])
        game = Model.Game(home, away)
        pred = engine.predict(game)
        market_odds = Prediction.MarketOdds.from_floats(row["home_decimal"], row["away_decimal"], row["tie_decimal"])
        #NOTE: De måste ha kört ett antal gånger först
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
            dates.append(row["Date"])
    # print(f"{correct}/{total} = {correct/total}")

    print(engine.state)
    print(len(b_lst))

    fig, ax = plt.subplots()
    ax.plot(dates, b_lst)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    # ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()

    plt.show()


    # teams = [s.teams for s in state_history]
    # for team in state.teams.keys():
    #     plt.plot(dates, [t[team].attack.mean for t in teams], label=team)
    # plt.legend()
    # plt.show()

test()
