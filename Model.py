#!/bin/python3
from __future__ import annotations
from enum import Enum
from typing import TypeVar, Generic
from dataclasses import dataclass
import numpy as np
from scipy.stats import truncnorm, norm


DEFAULT_START_MU = 25
DEFAULT_START_VAR = (25 / 3) ** 2
DEFAULT_VAR_T = (25 / 6) ** 2


T = TypeVar("T")  # Team skill type, advanced, general,..


@dataclass
class PosteriorResult(Generic[T]):
    home_posterior: T
    away_posterior: T
    home_advantage_posterior: NormalDistribution | None = None
    def __str__(self):
        s = f"Home: {self.home_posterior}\nAway: {self.away_posterior}"
        s = s + f"\nHome Advantage: {self.home_advantage_posterior}" if self.home_advantage_posterior else ""
        return s


@dataclass
class PredictionResult:
    p_home: float
    p_away: float
    p_tie: float
    def __str__(self):
        return f"{self.p_home}, {self.p_tie}, {self.p_away}"


class Outcome(Enum):
    HOME_WIN = 1
    AWAY_WIN = 2
    TIE = 3


@dataclass
class Score:
    home_goals: int
    away_goals: int

    @property
    def outcome(self):
        if self.home_goals > self.away_goals:
            return Outcome.HOME_WIN
        elif self.home_goals < self.away_goals:
            return Outcome.AWAY_WIN
        else:
            return Outcome.TIE
    def __str__(self):
        return f"{self.home_goals} home goals\n {self.away_goals} away goals"


@dataclass
class Game:
    home: str
    away: str
    score: Score | None = None
    def __str__(self):
        return f"{self.home}/{self.away}" + (f"- {self.score}" if self.score else "")



@dataclass
class AttackDefenseSkill:
    attack: NormalDistribution
    defense: NormalDistribution
    def __str__(self):
        return f"Attack {self.attack} & Defense {self.defense}"


@dataclass
class GeneralSkill:
    skill: NormalDistribution
    def __str__(self):
        return f"General skill {self.skill}"


@dataclass
class ModelState(Generic[T]):
    teams: dict[str, T]
    home_advantage: NormalDistribution | None = None
    def __str__(self):
        return f"Home Advantage ---> {self.home_advantage} <--- Home Advantage\n" + "\n".join([f"{v} <--- {k}" for k, v in self.teams.items()])

def moment_match_truncated_gauss(start, end, m0, s0):
    a_scaled, b_scaled = (start - m0) / np.sqrt(s0), (end - m0) / np.sqrt(s0)
    m = truncnorm.mean(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
    s = truncnorm.var(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
    return m, s


def get_gaussian_density_function(mean, variance, start, stop):
    x = np.linspace(start, stop, 100)
    curve = norm.pdf(x, loc=mean, scale=np.sqrt(variance))
    return x, curve


def get_cumulative_distribution_function(x1, x2, mean=0, var=1):
    if x1 > x2:
        raise ValueError(f"{x1=}<{x2=}")
    return norm.cdf(x2, mean, np.sqrt(var)) - norm.cdf(x1, mean, np.sqrt(var))


def get_gaussian_density_function_for_plotting(mean, variance):
    var_3 = 3 * np.sqrt(variance)
    return get_gaussian_density_function(mean, variance, mean - var_3, mean + var_3)



class TrueskillAdapter(Generic[T]):
    def get_posterior(
        self, game: Game, state: ModelState[T]
    ) -> tuple[ModelState[T], PosteriorResult[T]]:
        raise NotImplementedError

    # TODO: fixa return type
    def get_probabilities(self, game: Game, state: ModelState[T]) -> PredictionResult:
        raise NotImplementedError

class GameEngine(Generic[T]):
    def __init__(self, adapter:TrueskillAdapter[T], state: ModelState[T]):
        self.adapter = adapter  # Easily change adapter during runtime
        self.state = state

    def update(
        self,
        game: Game,
    ) -> PosteriorResult[T]:
        new_state, result = self.adapter.get_posterior(game, self.state)
        self.state = new_state
        return result

    def predict(self, game: Game) -> PredictionResult:
        return self.adapter.get_probabilities(game, self.state)


class TrueskillBasicAdapter(TrueskillAdapter[GeneralSkill]):
    def __init__(self, model): # borde ha type här
        self.model = model

    def get_posterior(
        self, game: Game, state: ModelState[GeneralSkill]
    ) -> tuple[ModelState[GeneralSkill], PosteriorResult[GeneralSkill]]:
        if not isinstance(game.score, Score):
            raise ValueError("Please enter a game with a score for training")
        outcome = game.score.outcome
        home = state.teams[game.home]
        away = state.teams[game.away]
        if state.home_advantage:
            raise ValueError("Trueskill basic does not implement home advantage....yet")
        # if type(home) != GeneralSkill or type(away) != GeneralSkill:
        #     raise ValueError("Please use GeneralSkill for this model")

        winner, loser = None, None
        if outcome == Outcome.HOME_WIN:
            winner, loser = home, away
        elif outcome == Outcome.AWAY_WIN:
            winner, loser = away, home
        else:
            raise ValueError("Trueskill basic does not implement ties")

        winner_post, loser_post = self.model.get_posterior(winner.skill, loser.skill)
        if outcome == Outcome.HOME_WIN:
            home_post, away_post = winner_post, loser_post
        else:
            away_post, home_post = winner_post, loser_post 

        new_state = ModelState(
        teams={
            **state.teams,
            game.home: GeneralSkill(skill=home_post),
            game.away: GeneralSkill(skill=away_post),
        },
    )
        posterior_result = PosteriorResult(
            home_posterior=GeneralSkill(skill=home_post),
            away_posterior=GeneralSkill(skill=away_post)
        )
        return new_state, posterior_result

    # TODO: fixa modelstate (string till skills)
    def get_probabilities(self, game: Game, state: ModelState[GeneralSkill]) -> PredictionResult:
        score = game.score
        home = state.teams[game.home]
        away = state.teams[game.away]
        if score is not None:
            raise ValueError("Cannot process outcome for probability calculation")
        p_home, p_away = self.model.get_probablities(home.skill, away.skill)
        return PredictionResult(p_home, p_away, 0)


class TrueskillAdvancedAdapter(TrueskillAdapter[GeneralSkill]):
    """
    adaptrarna är gränsen för fina gränssnitts-grunkor (game, generalskill etc).
    NormalDist-grunkan är ok däremot, används faktiskt i matten
    """

    def _convert_game_outcome(self, outcome: Outcome):
        if outcome == Outcome.HOME_WIN:
            return 0
        elif outcome == Outcome.TIE:
            return 1
        elif outcome == Outcome.AWAY_WIN:
            return 2

    def __init__(self, model):
        self.model = model

    def get_posterior(
        self, game: Game, state: ModelState[GeneralSkill]
    ) -> tuple[ModelState[GeneralSkill], PosteriorResult[GeneralSkill]]:
        if not isinstance(game.score, Score):
            raise ValueError("Please enter a game with a score for training")
        home = state.teams[game.home]
        away = state.teams[game.away]
        if not state.home_advantage:
            raise ValueError("This model needs a home_advantage parameter")
        home_post, away_post, home_advantage_post = self.model.get_posterior(
            home.skill,
            away.skill,
            state.home_advantage,
            self._convert_game_outcome(game.score.outcome),
        )

        new_state = ModelState(
            teams={
                **state.teams,
                game.home: GeneralSkill(skill=home_post),
                game.away: GeneralSkill(skill=away_post)
            },
                home_advantage=home_advantage_post
        )
        posterior_result = PosteriorResult(
            home_posterior=GeneralSkill(skill=home_post),
            away_posterior = GeneralSkill(skill=away_post),
            home_advantage_posterior=home_advantage_post,
        )
        return new_state, posterior_result

    def get_probabilities(self, game: Game, state: ModelState[GeneralSkill]) -> PredictionResult:
        score = game.score
        home = state.teams[game.home]
        away = state.teams[game.away]
        if score is not None:
            raise ValueError("Cannot process outcome for probability calculation")
        if not state.home_advantage:
            raise ValueError("This model needs a home_advantage parameter")
        p_home, p_away, p_tie = self.model.get_probablities(
            home.skill, away.skill, state.home_advantage
        )
        return PredictionResult(p_home, p_away, p_tie)


class TrueskillAdvancedAttackDefenseAdapter(TrueskillAdapter[AttackDefenseSkill]):
    def __init__(self, model:TrueskillAdvancedAttackDefense):
        self.model = model

    def get_posterior(
        self, game: Game, state: ModelState[AttackDefenseSkill]
    ) -> tuple[ModelState[AttackDefenseSkill], PosteriorResult[AttackDefenseSkill]]:
        if not isinstance(game.score, Score):
            raise ValueError("Please enter a game with a score for training")
        home_goals = game.score.home_goals
        away_goals = game.score.away_goals
        home = state.teams[game.home]
        away = state.teams[game.away]
        home_attack = home.attack
        home_defense = home.defense
        away_attack = away.attack
        away_defense = away.defense
        if not state.home_advantage:
            raise ValueError("This model needs a home_advantage parameter")
        (
            home_attack_post,
            home_defense_post,
            away_attack_post,
            away_defense_post,
            home_advantage_post,
        ) = self.model.get_posterior(
            home_attack,
            home_defense,
            away_attack,
            away_defense,
            state.home_advantage,
            home_goals,
            away_goals,
        )
        new_state = ModelState(
            teams = {
                **state.teams,
                game.home: AttackDefenseSkill(attack=home_attack_post, defense=home_defense_post),
                game.away: AttackDefenseSkill(attack=away_attack_post, defense=away_defense_post),
            },
                home_advantage=home_advantage_post
        )
        posterior_result = PosteriorResult(
            home_posterior = AttackDefenseSkill(attack=home_attack_post, defense=home_defense_post),
            away_posterior = AttackDefenseSkill(attack=away_attack_post, defense=away_defense_post),
            home_advantage_posterior = home_advantage_post,
        )
        return new_state, posterior_result

    def get_probabilities(self, game: Game, state: ModelState[AttackDefenseSkill]) -> PredictionResult:
        score = game.score
        home = state.teams[game.home]
        away = state.teams[game.away]
        home_attack = home.attack
        home_defense = home.defense
        away_attack = away.attack
        away_defense = away.defense
        if score is not None:
            raise ValueError("Cannot process outcome for probability calculation")
        if not state.home_advantage:
            raise ValueError("This model needs a home_advantage parameter")
        p_home, p_away, p_tie = self.model.get_probabilities(
            home_attack, home_defense, away_attack, away_defense, state.home_advantage
        )
        return PredictionResult(p_home, p_away, p_tie)


class TrueSkillModelBasic:
    def __init__(self, var_t: float):
        self.var_t = var_t

    def _get_probablity_home_wins(
        self, home: NormalDistribution, away: NormalDistribution
    ) -> float:
        t_given_skills = self._get_marginal_t_distribution(home, away)
        return t_given_skills.cumulative_probability(0, np.inf)

    def _get_marginal_t_distribution(
        self, home: NormalDistribution, away: NormalDistribution
    ) -> NormalDistribution:
        """Marginal distribution of t, tror här att home antar vinnare"""
        return NormalDistribution(
            home.mean - away.mean, self.var_t + home.variance + away.variance
        )

    # ------------------------------Gibbs sampling methods------------------------------
    def draw_sample_t_given_all(self, winner_sample: float, loser_sample: float):
        a_trunc = 0
        scale = np.sqrt(self.var_t)
        loc = winner_sample - loser_sample
        a = (a_trunc - loc) / scale
        return truncnorm.rvs(a=a, b=np.inf, loc=loc, scale=scale)

    def draw_sample_winner_given_all(self, winner, t_sample, loser_sample):
        loc = (
            (t_sample + loser_sample) * winner.variance + winner.mean * self.var_t
        ) / (self.var_t + winner.variance)
        scale = (winner.mean * self.var_t) / (winner.variance + self.var_t)
        scale = np.sqrt(scale)
        return np.random.normal(loc, scale)

    def draw_sample_loser_given_all(self, loser, t_sample, winner_sample):
        loc = (
            (winner_sample - t_sample) * loser.variance + loser.mean * self.var_t
        ) / (self.var_t + loser.variance)
        scale = (loser.variance * self.var_t) / (loser.variance + self.var_t)
        scale = np.sqrt(scale)
        return np.random.normal(loc, scale)

    # ------------------------------Message passing methods------------------------------
    def get_t_hat_given_winner(self, winner, loser):
        """
        Assumes observed winner
        P(t|y=1), which is a truncated gaussian.
        Perfom moment matching to achieve a somewhat accurate gaussian (approximation)
        """
        t_hat = self._get_marginal_t_distribution(
            winner, loser
        ).truncate_and_moment_match(0, np.inf)
        return t_hat

    def get_message_to_t_from_s(self, winner, loser):
        """
        Assumes observed winner
        Message from the common s1,s2,t factor towards t
        """
        message_to_t_from_s = NormalDistribution(
            winner.mean - loser.mean,
            winner.variance + loser.variance + self.var_t,
        )
        return message_to_t_from_s

    def get_message_to_s_from_t(self, t_hat, message_to_t_from_s):
        """
        Assumes observed winner
        Message from t towards the common s1,s2,t factor
        """
        message_from_t_hat_to_s = t_hat / message_to_t_from_s
        return message_from_t_hat_to_s

    def get_message_from_t_to_winner(self, message_from_t_hat_to_s, loser):
        """
        Assumes observed winner
        Message from the common factor s1,s2,t towards winner
        """
        message_from_t_hat_to_winner = NormalDistribution(
            message_from_t_hat_to_s.mean + loser.mean,
            self.var_t + loser.variance + message_from_t_hat_to_s.variance,
        )
        return message_from_t_hat_to_winner

    def get_message_from_t_to_loser(self, message_from_t_hat_to_s, winner):
        """
        Assumes observed winner
        Message from the common factor s1,s2,t towards loser
        """
        message_from_t_hat_to_loser = NormalDistribution(
            -message_from_t_hat_to_s.mean + winner.mean,
            self.var_t + winner.variance + message_from_t_hat_to_s.variance,
        )
        return message_from_t_hat_to_loser

    def get_probabilities(
        self, home: NormalDistribution, away: NormalDistribution
    ) -> tuple[float, float, float]:
        p_home = self._get_probablity_home_wins(home, away)
        p_away = 1 - p_home
        p_tie = 0
        assert 1 - (p_home + p_away + p_tie) < 0.0001
        return p_home, p_tie, p_away


class TrueSkillBasicGibbsSampling:
    """
    Class implementing Gibbs sampling for TrueSkill
    Currently limited to the model class TrueSkill-basic but could be generalized to handle any model
    """
    def __init__(self, model, burn_in_time, num_t):
        self.model = model
        self.burn_in_time = burn_in_time
        self.num_t = num_t

    # fix later
    # @staticmethod
    # def getHistOfDifferentT(var_t, mu_home, var_home, mu_away, var_away, t_range=1000):
    #     """
    #     Detta ska bli en classmetod för att kalibrera en lagom gibbs T. Flytta dit alltså
    #     Plots gibbs hist together with fitted gauss in order to find a good T
    #     """
    #     sample_series = TrueSkillBasicGibbsSampling._gibbs_init(
    #         mu_home, var_home, mu_away, var_away, var_t, t_range
    #     )
    #     sample_series = TrueSkillBasicGibbsSampling._gibbs_run_samples(
    #         sample_series, mu_home, var_home, mu_away, var_away, var_t, t_range
    #     )
    #     plt.figure(1)
    #     plt.plot(sample_series, label=("home", "away", "t"))
    #     # plt.hist(sample_series, label=("home", "away", "t"))
    #     plt.grid()
    #     plt.legend()
    #     plt.show()


    def _extract_posterior_from_series(self, sample_series):
        winner = sample_series[:, 0]
        loser = sample_series[:, 1]
        winner = NormalDistribution(np.mean(winner), np.var(winner))
        loser = NormalDistribution(np.mean(loser), np.var(loser))
        return winner, loser

    def _post_process_sample_series(self, sample_series):
        return sample_series[self.burn_in_time :, :]

    def _gibbs_run_samples(self, x, winner, loser):
        for t in range(self.num_t):
            x[t + 1, :] = self._gibbs_run_one_sample(x[t, :], winner, loser)
        return x

    def _gibbs_run_one_sample(self, state, winner, loser):
        new_state = []
        new_state.append(
            self.model.draw_sample_winner_given_all(
                # state[2], state[1], winner, var_t
                winner,
                state[2],
                state[1],
            )
        )
        new_state.append(
            self.model.draw_sample_loser_given_all(
                # state[2], new_state[0], loser, var_t
                loser,
                state[2],
                new_state[0],
            )
        )
        new_state.append(self.model.draw_sample_t_given_all(new_state[0], new_state[1]))
        return new_state

    def _gibbs_init(self, winner, loser):
        x = np.zeros(shape=(self.num_t + 1, 3))
        # x[0, :] = draw_from_normal(
        #     ((winner.mean, winner.variance), (loser.mean, loser.variance), (winner.mean - loser.mean, var_t))
        # )
        x[0, :] = [winner.mean, loser.mean, winner.mean - loser.mean]
        return x

    def get_posterior(
        self,
        winner: NormalDistribution,
        loser: NormalDistribution,
    ) -> tuple[NormalDistribution, NormalDistribution]:
        # model,
        # t_range=1000,
        # burn_in_time=50,

        sample_series = self._gibbs_init(winner, loser)
        sample_series = self._gibbs_run_samples(sample_series, winner, loser)
        sample_series = self._post_process_sample_series(sample_series)
        winner_post, loser_post = self._extract_posterior_from_series(sample_series)
        return winner_post, loser_post

    def get_probabilities(self, home, away):
        return self.model.get_probabilities(home, away)


class TrueSkillBasicMessagePassing:
    """
    Class implementing message passing for TrueSkill
    """

    @staticmethod
    def _get_posterior(model: TrueSkillModelBasic, winner, loser):
        message_to_t_from_s = model.get_message_to_t_from_s(winner, loser)
        t_hat = model.get_t_hat_given_winner(winner, loser)
        message_from_t_hat_to_s = model.get_message_to_s_from_t(
            t_hat, message_to_t_from_s
        )
        winner_post = (
            model.get_message_from_t_to_winner(message_from_t_hat_to_s, loser) * winner
        )
        loser_post = (
            model.get_message_from_t_to_loser(message_from_t_hat_to_s, winner) * loser
        )

        # t_hat = NormalDistribution(
        #     winner.mean - loser.mean, winner.variance + loser.variance + var_t
        # ).truncate_and_moment_match(0, np.inf)
        # message_to_t_from_s = NormalDistribution(
        #     winner.mean - loser.mean, winner.variance + loser.variance + var_t
        # )
        # message_from_t_hat_to_s = t_hat / message_to_t_from_s
        # message_from_t_hat_to_winner = NormalDistribution(
        #     message_from_t_hat_to_s.mean + loser.mean,
        #     var_t + loser.variance + message_from_t_hat_to_s.variance,
        # )
        # message_from_t_hat_to_loser = NormalDistribution(
        #     -message_from_t_hat_to_s.mean + winner.mean,
        #     var_t + winner.variance + message_from_t_hat_to_s.variance,
        # )
        # winner_post = home * message_from_t_hat_to_winner
        # loser_post = loser * message_from_t_hat_to_loser

        # tHat = moment_match_truncated_gauss(
        #     0, np.inf, mu_home - mu_away, var_home + var_away + var_t
        # )
        # mu_4 = [mu_home - mu_away, var_home + var_away + var_t]
        # tMessage = divide_gauss(tHat[0], tHat[1], mu_4[0], mu_4[1])
        # home = multiply_gauss(
        #     mu_home,
        #     var_home,
        #     tMessage[0] + mu_away,
        #     var_t + var_away + tMessage[1],
        # )
        # away = multiply_gauss(
        #     mu_away,
        #     var_away,
        #     -tMessage[0] + mu_home,
        #     var_t + var_home + tMessage[1],
        # )
        # return winner_post, loser_post
        return winner_post, loser_post
    def get_probabilities(self, home, away):
        return self.model.get_probabilities(home, away)


class TrueskillAdvancedAttackDefense:
    """Implements trueskill with tie functionality, home team advantage and attack-defense logic"""

    def __init__(self, var_t, var_y):
        self.var_t = var_t
        self.var_y = var_y

    def _home_attack_posterior(
        self,
        home_attack: NormalDistribution,
        away_defense: NormalDistribution,
        home_advantage: NormalDistribution,
        home_goals: int,
    ) -> NormalDistribution:
        A = np.array([1, -1, 1]).reshape([1, -1])
        b = 0
        cov_ba = self.var_t
        mu_a = np.array(
            [away_defense.mean, home_advantage.mean, home_goals],
        ).reshape([-1, 1])
        cov_a = np.diag([away_defense.variance, home_advantage.variance, self.var_y])
        return home_attack * marginal_distribution(A, b, cov_ba, mu_a, cov_a)

    def _away_defense_posterior(
        self,
        home_attack: NormalDistribution,
        away_defense: NormalDistribution,
        home_advantage: NormalDistribution,
        home_goals: int,
    ) -> NormalDistribution:
        A = np.array([1, 1, -1]).reshape([1, -1])
        b = 0
        cov_ba = self.var_t
        mu_a = np.array(
            [home_attack.mean, home_advantage.mean, home_goals],
        ).reshape([-1, 1])
        cov_a = np.diag([home_attack.variance, home_advantage.variance, self.var_y])
        return away_defense * marginal_distribution(A, b, cov_ba, mu_a, cov_a)

    def _home_defense_posterior(
        self,
        home_defense: NormalDistribution,
        away_attack: NormalDistribution,
        home_advantage: NormalDistribution,
        away_goals: int,
    ) -> NormalDistribution:
        A = np.array([1, -1, -1]).reshape([1, -1])
        b = 0
        cov_ba = self.var_t
        mu_a = np.array(
            [away_attack.mean, home_advantage.mean, away_goals],
        ).reshape([-1, 1])
        cov_a = np.diag([away_attack.variance, home_advantage.variance, self.var_y])
        return home_defense * marginal_distribution(A, b, cov_ba, mu_a, cov_a)

    def _away_attack_posterior(
        self,
        home_defense: NormalDistribution,
        away_attack: NormalDistribution,
        home_advantage: NormalDistribution,
        away_goals: int,
    ) -> NormalDistribution:
        A = np.array([1, 1, 1]).reshape([1, -1])
        b = 0
        cov_ba = self.var_t
        mu_a = np.array(
            [home_defense.mean, home_advantage.mean, away_goals],
        ).reshape([-1, 1])
        cov_a = np.diag([home_defense.variance, home_advantage.variance, self.var_y])
        return away_attack * marginal_distribution(A, b, cov_ba, mu_a, cov_a)

    def get_posterior(
        self,
        home_attack: NormalDistribution,
        home_defense: NormalDistribution,
        away_attack: NormalDistribution,
        away_defense: NormalDistribution,
        home_advantage: NormalDistribution,
        home_goals: int,
        away_goals: int,
    ) -> tuple[NormalDistribution, NormalDistribution, NormalDistribution, NormalDistribution, NormalDistribution]:
        # en liten fuling med home_advantage - borde vara parallell update egentligen
        home_attack_new = self._home_attack_posterior(
            home_attack, away_defense, home_advantage, home_goals
        )
        away_defense_new = self._away_defense_posterior(
            home_attack, away_defense, home_advantage, home_goals
        )
        home_advantage_new = self._home_attack_posterior(
            home_advantage, away_defense, home_attack, home_goals
        )

        home_defense_new = self._home_defense_posterior(
            home_defense, away_attack, home_advantage, away_goals
        )
        away_attack_new = self._away_attack_posterior(
            home_defense, away_attack, home_advantage, away_goals
        )
        home_advantage_new = self._home_defense_posterior(
            home_advantage_new, away_attack, home_defense, away_goals
        )
        return (
            home_attack_new,
            home_defense_new,
            away_attack_new,
            away_defense_new,
            home_advantage_new,
        )

    def get_probabilities(self, home_attack, home_defense, away_attack, away_defense, home_advantage):
        home_goals_marginal_y = marginal_distribution(1, 0, self.var_y, home_attack.mean-away_defense.mean+home_advantage.mean, self.var_t)
        away_goals_marginal_y = marginal_distribution(1, 0, self.var_y, away_attack.mean-home_defense.mean-home_advantage.mean, self.var_t)
        diff = home_goals_marginal_y - away_goals_marginal_y
        p_away = diff.cumulative_probability(-np.inf, -0.5)
        p_tie = diff.cumulative_probability(-0.5, 0.5)
        p_home = diff.cumulative_probability(0.5, np.inf)
        return p_home, p_away, p_tie

def kelly_basic(prob, payout):
    return 0.2 * (prob * (payout + 1) - 1) / payout


def multiply_gauss(m1, s1, m2, s2):
    m = (m1 * s2 + m2 * s1) / (s1 + s2)
    s = s1 * s2 / (s1 + s2)
    return [m, s]


def divide_gauss(m1, s1, m2, s2):
    m = (m1 * s2 - m2 * s1) / (s2 - s1)
    s = s1 * s2 / (s2 - s1)
    return [m, s]


def marginal_distribution(A, b, cov_ba, mu_a, cov_a):
    """
    p(x_a) = N(x_a; mu_a, cov_a)
    p(x_b | x_a) = N(x_b; A*x_a + b, cov_b|a)
    Returns:
        p(x_b) = N(x_b; mu_b, cov_b)
    """
    A = np.array(A)
    b = np.array(b)
    cov_ba = np.array(cov_ba)
    cov_a = np.array(cov_a)
    mu_a = np.array(mu_a)
    try:
        mu_b = A @ mu_a + b
        cov_b = cov_ba + A @ cov_a @ A.T
    except ValueError:
        mu_b = A * mu_a + b
        cov_b = cov_ba + A * cov_a * A.T
    try:
        return NormalDistribution(mu_b.item(), cov_b.item())
    except ValueError:
        raise NotImplementedError("ERROR: No support for multidimensional distributions for p(x_b)")
        # return mu_b, cov_b


class TrueskillAdvanced:
    """Implements truskill with tie functionality and home team advantage"""

    def __init__(
        self,
        var_t: float,
        tie_epsilon: float,
    ):
        self.var_t = var_t
        self.tie_epsilon = tie_epsilon

    def _get_probablity_home_wins(
        self,
        t_marginal: NormalDistribution,
        home: NormalDistribution,
        away: NormalDistribution,
    ):
        return t_marginal.cumulative_probability(self.tie_epsilon, np.inf)

    def _get_probablity_tie(
        self,
        t_marginal: NormalDistribution,
        home: NormalDistribution,
        away: NormalDistribution,
    ):
        return t_marginal.cumulative_probability(-self.tie_epsilon, self.tie_epsilon)

    def _get_t_marginal_distribution(
        self,
        home: NormalDistribution,
        away: NormalDistribution,
        home_advantage: NormalDistribution,
    ) -> NormalDistribution:
        """Marginal distribution of t with nothing observed"""
        return NormalDistribution(
            home.mean - away.mean + home_advantage.mean,
            self.var_t + home.variance + away.variance + home_advantage.variance,
        )

    def _get_t_hat_given_winner(
        self,
        t_marginal: NormalDistribution,
    ) -> NormalDistribution:
        return t_marginal.truncate_and_moment_match(self.tie_epsilon, np.inf)

    def _get_t_hat_given_loser(
        self,
        t_marginal: NormalDistribution,
    ) -> NormalDistribution:
        return t_marginal.truncate_and_moment_match(-np.inf, -self.tie_epsilon)

    def _get_t_hat_given_tie(
        self,
        t_marginal: NormalDistribution,
    ) -> NormalDistribution:
        return t_marginal.truncate_and_moment_match(-self.tie_epsilon, self.tie_epsilon)

    def _get_home_post(
        self,
        home: NormalDistribution,
        away: NormalDistribution,
        home_advantage: NormalDistribution,
        t_hat_msg: NormalDistribution,
    ) -> NormalDistribution:
        denom = home.variance + self.var_t
        A = home.variance / (denom) * np.array([1, 1, -1]).reshape([1, -1])
        b = home.mean * self.var_t / (denom)
        cov_ba = home.variance * self.var_t / (denom)
        mu_a = np.array(
            [t_hat_msg.mean, home_advantage.mean, away.mean],
        ).reshape([-1, 1])
        cov_a = np.diag([t_hat_msg.variance, home_advantage.variance, away.variance])
        return marginal_distribution(A, b, cov_ba, mu_a, cov_a)

    def _get_home_post2(
        self,
        home: NormalDistribution,
        away: NormalDistribution,
        home_advantage: NormalDistribution,
        t_hat_msg: NormalDistribution,
    ) -> NormalDistribution:
        A = np.array([1, 1, -1]).reshape([1, -1])
        b = 0
        cov_ba = self.var_t
        mu_a = np.array([t_hat_msg.mean, away.mean, home_advantage.mean]).reshape(
            [-1, 1]
        )
        cov_a = np.diag([t_hat_msg.variance, away.variance, home_advantage.variance])
        return home * marginal_distribution(A, b, cov_ba, mu_a, cov_a)

    def _get_away_post(
        self,
        home: NormalDistribution,
        away: NormalDistribution,
        home_advantage: NormalDistribution,
        t_hat_msg: NormalDistribution,
    ) -> NormalDistribution:
        denom = away.variance + self.var_t
        A = away.variance / (denom) * np.array([1, 1, -1]).reshape([1, -1])
        b = away.mean * self.var_t / (denom)
        cov_ba = away.variance * self.var_t / (denom)
        mu_a = np.array(
            [home.mean, home_advantage.mean, t_hat_msg.mean],
        ).reshape([-1, 1])
        cov_a = np.diag([home.variance, home_advantage.variance, t_hat_msg.variance])
        return marginal_distribution(A, b, cov_ba, mu_a, cov_a)

    def _get_away_post2(
        self,
        home: NormalDistribution,
        away: NormalDistribution,
        home_advantage: NormalDistribution,
        t_hat_msg: NormalDistribution,
    ) -> NormalDistribution:
        A = np.array([1, -1, 1]).reshape([1, -1])
        b = 0
        cov_ba = self.var_t
        mu_a = np.array([home.mean, t_hat_msg.mean, home_advantage.mean]).reshape(
            [-1, 1]
        )
        cov_a = np.diag([home.variance, t_hat_msg.variance, home_advantage.variance])
        return away * marginal_distribution(A, b, cov_ba, mu_a, cov_a)

    def _get_home_advantage_post(
        self,
        home: NormalDistribution,
        away: NormalDistribution,
        home_advantage: NormalDistribution,
        t_hat_msg: NormalDistribution,
    ) -> NormalDistribution:
        return self._get_home_post(home_advantage, away, home, t_hat_msg)

    def _get_home_advantage_post2(
        self,
        home: NormalDistribution,
        away: NormalDistribution,
        home_advantage: NormalDistribution,
        t_hat_msg: NormalDistribution,
    ) -> NormalDistribution:
        return self._get_home_post2(home_advantage, away, home, t_hat_msg)

    def _get_msg_node_t_to_factor_t(
        self, t_observed_hat: NormalDistribution, t_marginal: NormalDistribution
    ) -> NormalDistribution:
        return t_observed_hat / t_marginal

    def get_posterior(
        self,
        home: NormalDistribution,
        away: NormalDistribution,
        home_advantage: NormalDistribution,
        winner_index: int,
    ) -> tuple[NormalDistribution, NormalDistribution, NormalDistribution]:
        """
        winner index:
            0 home wins
            1 tie
            2 away wins
        """
        t_marginal = self._get_t_marginal_distribution(home, away, home_advantage)
        t_observed_hat = [
            self._get_t_hat_given_winner,
            self._get_t_hat_given_tie,
            self._get_t_hat_given_loser,
        ][winner_index](t_marginal)

        t_hat_msg = self._get_msg_node_t_to_factor_t(t_observed_hat, t_marginal)
        home_post2 = self._get_home_post2(home, away, home_advantage, t_hat_msg)
        away_post2 = self._get_away_post2(home, away, home_advantage, t_hat_msg)
        home_advantage_post2 = self._get_home_advantage_post2(
            home, away, home_advantage, t_hat_msg
        )
        return home_post2, away_post2, home_advantage_post2

    def get_probablities(
        self,
        home: NormalDistribution,
        away: NormalDistribution,
        home_advantage: NormalDistribution,
    ) -> tuple[NormalDistribution, NormalDistribution, NormalDistribution]:
        t_marginal = self._get_t_marginal_distribution(home, away, home_advantage)
        p_home = self._get_probablity_home_wins(t_marginal, home, away)
        p_tie = self._get_probablity_home_wins(t_marginal, home, away)
        p_away = 1 - p_tie - p_home
        assert 1 - (p_home + p_away + p_tie) < 0.0001
        return p_home, p_tie, p_away




@dataclass(frozen=True) # avoid weird shared instance errors
class NormalDistribution:
    """
    Class for commonly used operations for normal distributions
    """
    mean: float
    variance: float

    # def __init__(self, mean:float, variance:float):
        # self.mean = mean
        # self.variance = variance

    def __str__(self):
        return f"N({round(self.mean,2)}, {round(self.variance,2)})"

    def __mul__(self, other: NormalDistribution) -> NormalDistribution:
        return NormalDistribution(
            *multiply_gauss(self.mean, self.variance, other.mean, other.variance)
        )

    def __truediv__(self, other: NormalDistribution) -> NormalDistribution:
        return NormalDistribution(
            *divide_gauss(self.mean, self.variance, other.mean, other.variance)
        )
    def __add__(self, other:NormalDistribution):
        roe = 0 # add seperate for dependent variables
        return NormalDistribution(self.mean+other.mean, self.variance + other.variance + 2*roe*np.sqrt(self.variance)*np.sqrt(other.variance))

    def __sub__(self, other:NormalDistribution):
        roe = 0 # add seperate for dependent variables
        return NormalDistribution(self.mean-other.mean, self.variance + other.variance - 2*roe*np.sqrt(self.variance)*np.sqrt(other.variance))

    def truncate_and_moment_match(self, start: float, end: float) -> NormalDistribution:
        """Truncate gaussian and perform moment matching"""
        return NormalDistribution(
            *moment_match_truncated_gauss(start, end, self.mean, self.variance)
        )

    def density_function(
        self, start=None, stop=None, num=100
    ) -> tuple[float, list[float]]:
        if start is None:
            start = self.mean - 3 * np.sqrt(self.variance)
        if stop is None:
            stop = self.mean + 3 * np.sqrt(self.variance)
        return get_gaussian_density_function(self.mean, self.variance, start, stop)

    def cumulative_probability(self, start: float, stop: float) -> float:
        return float(get_cumulative_distribution_function(
            start, stop, self.mean, self.variance
        ))

    def draw_sample(self):
        return np.random.normal(self.mean, np.sqrt(self.variance))

"""
def compare_gibbs_and_message():
    init_mu = 25
    init_var = np.power(25 / 3, 2)
    var_t = np.power(25 / 6, 2)
    samples = 1000
    results = TrueSkillBasicGibbsSampling._get_posterior(
        var_t, init_mu, init_var, init_mu, init_var, samples, 50
    )
    results2 = TrueSkillBasicMessagePassing._get_posterior(
        var_t, init_mu, init_var, init_mu, init_var
    )
    print(f"s1: mean {results[0]}, var {results[1]}")
    print(f"s2: mean {results[2]}, var {results[3]}")
    print(f"s1: mean {results2[0]}, var {results2[1]}")
    print(f"s2: mean {results2[2]}, var {results2[3]}")
    # plot_grunka(results, results2)
    plot_results(
        (results, results2),
        "Model comparison",
        (f"Gibbs sampling {samples} samples", "Message passing"),
    )


def plot_results(team_attributes_lst, title, types):
    num_plots = len(team_attributes_lst)
    fig, ax_lst = plt.subplots(num_plots)
    fig.suptitle(title)
    for i, team_attribute in enumerate(team_attributes_lst):
        ax = ax_lst[i]
        mean_s1, var_s1, mean_s2, var_s2 = team_attribute
        x, s1_curve = get_gaussian_density_function_for_plotting(
            mean_s1,
            var_s1,
        )
        ax.plot(
            x, s1_curve, label=f"Skill team 1 | ({round(mean_s1)}, {round(var_s1)})"
        )
        x, s2_curve = get_gaussian_density_function_for_plotting(
            mean_s2,
            var_s2,
        )
        ax.plot(
            x, s2_curve, label=f"Skill team 2 | ({round(mean_s2)}, {round(var_s2)})"
        )
        # maxHeightFit = np.max(s2_curve.extend(s1_curve))
        maxHeightFit = np.max(s2_curve)
        ax.legend()
        ax.set_ylim(0, maxHeightFit * 1.1)
        ax.set_title(types[i])
    plt.show()
"""


# def main():
#     init_mu = 25
# init_var = np.power(25 / 3, 2)
# var_t = np.power(25 / 6, 2)
# team_a = Team(init_mu, init_var)
# team_b = Team(init_mu, init_var)
# game_handler = GameHandler(
#     TrueSkillBasicGibbsSampling(burn_in_time=50, num_t=2000), var_t)
# )
# game_handler.update_team_skills_based_on_game(team_a, team_b)
# print(team_a.mean)
# print(team_a.variance)
# print(team_b.mean)
# print(team_b.variance)

# TrueSkillBasicGibbsSampling.getHistOfDifferentT(
#     var_t, init_mu, init_var, init_mu + 10, init_var, 1000
# )


# main()
# compare_gibbs_and_message()
