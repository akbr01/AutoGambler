#!/bin/python3
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import Model

def odds_expected_value(prob1, payout1, payout2):
    return prob1*payout1 + (1-prob1)*payout2

def odds_variance(prob1, payout1, payout2):
    expected_value_x2 = odds_expected_value(
        prob1, payout1**2, payout2**2
    )
    expected_value_x_exp2 = odds_expected_value(prob1, payout1, payout2) ** 2
    return expected_value_x2 - expected_value_x_exp2

class BetStrategy:
    def decide(self, pred:Model.PredictionResult, market_odds:MarketOdds)->list[Bet]:
        raise NotImplementedError

class BetScaler:
    # TODO: Maybe consider adding more general parameters here
    def scale(self, prob:float, payout:float)->float:
        raise NotImplementedError

class BetSide(Enum):
    HOME = "home"
    AWAY = "away"
    TIE = "tie"


@dataclass
class Bet:
    side: BetSide
    odds: Odds
    stake: float

@dataclass
class Odds:
    dec:float

    @classmethod
    def from_net(cls, net): # init class from net odds instead
        return cls(net+1)

    @property
    def net(self) -> float:
        return self.dec - 1

    @property
    def implied_probability(self) -> float:
        return 1/self.dec

    @property
    def break_even_probability(self) -> float:
        """Expected value = 0"""
        return 1/(1 + self.net)

@dataclass
class MarketOdds:
    home: Odds
    away: Odds
    tie: Odds

    @classmethod
    def from_floats(cls, home:float, away:float, tie:float):
        return cls(home=Odds(home), away=Odds(away), tie=Odds(tie))

    def implied_probabilities(self)->dict[str, float]:
        return {"home": self.home.implied_probability, "away": self.away.implied_probability, "tie": self.tie.implied_probability}

    def break_even_probablilities(self)->dict[str, float]:
        return {"home": self.home.break_even_probability, "away": self.away.break_even_probability, "tie": self.tie.break_even_probability}

    def bookmaker_margin_(self) -> float:
        return sum(imp for imp in self.implied_probabilities().values())


def evaluate_bet(bet: Bet, outcome: Model.Outcome)->float:
    if (bet.side == BetSide.HOME and outcome == Model.Outcome.HOME_WIN) or (bet.side == BetSide.AWAY and outcome == Model.Outcome.AWAY_WIN) or (bet.side == BetSide.TIE and outcome == Model.Outcome.TIE):
        return bet.stake * bet.odds.net
    else:
        return -bet.stake


class KellyBetScaler(BetScaler):
    def __init__(self, multiplier=0.2):
        self.multiplier = multiplier

    def scale(self, prob:float, payout:float) -> float:
        return self.multiplier * (prob * (payout + 1) - 1) / payout


class ValueBetStrategy(BetStrategy):
    def __init__(self, treshold:float, scaler:BetScaler):
        self.treshold = treshold
        self.scaler = scaler

    def decide(self, pred:Model.PredictionResult, market_odds:MarketOdds)->list[Bet]:
        exp_home = odds_expected_value(pred.p_home, market_odds.home.net, -1)
        exp_away = odds_expected_value(pred.p_away, market_odds.away.net, -1)
        exp_tie = odds_expected_value(pred.p_tie, market_odds.tie.net, -1)
        mapping = {BetSide.HOME: (pred.p_home, market_odds.home, exp_home), BetSide.AWAY: (pred.p_away, market_odds.away, exp_away), BetSide.TIE: (pred.p_tie, market_odds.tie, exp_tie)}
        bet_lst = []
        for side, (prob, odds, exp) in mapping.items():
            if exp > self.treshold:
                # print(f"{side=}, {prob=}, {odds=}, {exp=}")
                bet_size = self.scaler.scale(prob, odds.net)
                bet_lst.append(Bet(side, odds, bet_size))
        return bet_lst


# class PredictionAndGamble:
#     """
#     When predicting game winner we should incorporate the uncertainty in our posterior when computing likelihood
#
#     odds_type_1:
#         odds = 2
#             betta 1 unit och få 2 om vin, annars förlora 1
#     odds_type_2 (decimal):
#         odds = 2
#             få unit*odds
#             betta 1 unit och få 2 tillbaka (1 unit profit)
#     """
#
#     def __init__(self, bet_factor_function):
#         self.bet_factor_function = bet_factor_function
#         # insert odds-parameters here, like kelly stuff
#
#     def _odds_expected_value(self, payout_s1, payout_s2, prob_s1_wins):
#         """
#         Edge
#         Odds 2:1 for s1 ---> payout1=2, payout2=1
#         """
#         return payout_s1 * prob_s1_wins + payout_s2 * (1 - prob_s1_wins)
#
#     def _odds_variance(self, payout_s1, payout_s2, prob_s1_wins):
#         """hello"""
#         expected_value_x2 = self._odds_expected_value(
#             payout_s1**2, payout_s2**2, prob_s1_wins
#         )
#         expected_value_x_exp2 = np.power(
#             self._odds_expected_value(payout_s1, payout_s2, prob_s1_wins),
#             2,
#         )
#         return expected_value_x2 - expected_value_x_exp2
#
#
#     def _evaluate_odds(self, p_home_wins, decimal_odds_dict):
#         """
#         Returns:
#             Dict_market[Dict[Team]] = [expected_value, var]
#         """
#         bet_lst = defaultdict(dict)
#         for market, odds_home_tie_away in decimal_odds_dict.items():
#             net_odds_home, net_odds_tie, net_odds_away = [
#                 self._decimal_to_net_odds(odds) for odds in odds_home_tie_away
#             ]
#             expected_value_home = self._odds_expected_value(
#                 net_odds_home, 0, p_home_wins
#             )
#             expected_value_away = self._odds_expected_value(
#                 net_odds_away, 0, p_home_wins
#             )
#             var_home = self._odds_variance(net_odds_home, 0, p_home_wins)
#             var_away = self._odds_variance(net_odds_away, 0, p_home_wins)
#             bet_lst[market]["H"] = [expected_value_home, var_home]
#             bet_lst[market]["A"] = [expected_value_away, var_away]
#         return bet_lst
#
#     def _sort_bet_lst(self, bet_lst):
#         return sorted(
#             bet_lst.items(),
#             key=lambda market_team: max(
#                 exp_var[0] for team, exp_var in market_team[1].items()
#             ),
#         )
#
#     def find_most_profitable_bet(self, p_home_wins, decimal_odds_dict):
#         bet_lst = self._evaluate_odds(p_home_wins, decimal_odds_dict)
#         best_bet = self._sort_bet_lst(bet_lst)[-1]
#         market, teams = best_bet
#         team_values = sorted(teams.items(), key=lambda x: x[1][0])[-1]
#         team, values = team_values
#         bet_decimal = decimal_odds_dict[market][0 if team == "H" else 1]
#         return market, team, values, bet_decimal
#
#     def get_bet(self, budget, p_home_wins, decimal_odds_dict):
#         market, team, values, bet_decimal = self.find_most_profitable_bet(
#             p_home_wins, decimal_odds_dict
#         )
#         amount = budget * self.bet_factor_function(
#             p_home_wins, self._decimal_to_net_odds(bet_decimal)
#         )
#         return (
#             market,
#             team,
#             values,
#             bet_decimal,
#             amount,
#         )
#
#
