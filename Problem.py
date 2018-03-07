import numpy as np
import scr.FigureSupport as figureLibrary
from enum import Enum
import numpy as np
import scr.SamplePathClass as PathCls
import scr.StatisticalClasses as Stat



class Game(object):
    def __init__(self, id, prob_head):
        self._id = id
        self._rnd = np.random
        self._rnd.seed(id)
        self._probHead = prob_head  # probability of flipping a head
        self._countWins = 0  # number of wins, set to 0 to begin

    def simulate(self, n_of_flips):
        count_tails = 0  # number of consecutive tails so far, set to 0 to begin
        # flip the coin 20 times
        for i in range(n_of_flips):

            # in the case of flipping a heads
            if self._rnd.random_sample() < self._probHead:
                if count_tails >= 2:  # if the series is ..., T, T, H
                    self._countWins += 1  # increase the number of wins by 1
                count_tails = 0  # the tails counter needs to be reset to 0 because a heads was flipped

            # in the case of flipping a tails
            else:
                count_tails += 1  # increase tails count by one

    def get_reward(self):
        # calculate the reward from playing a single game
        return 100*self._countWins - 250

    def get_loss(self,n_of_flips):
        count_loss=0
        for i in range(n_of_flips):
            if 100*self._countWins-250 < 0:
                count_loss+=1
        return count_loss


class SetOfGames:
    def __init__(self, prob_head, n_games):
        self._gameRewards = [] # create an empty list where rewards will be stored
        self._loss=[]
        # simulate the games
        for n in range(n_games):
            # create a new game
            game = Game(id=n, prob_head=prob_head)
            # simulate the game with 20 flips
            game.simulate(20)
            # store the reward
            self._gameRewards.append(game.get_reward())
            self._loss.append(game.get_loss(20)/len(self._gameRewards))

    def simulation(self):
        return CohortOutcomes(self)

    def get_ave_reward(self):
        """ returns the average reward from all games"""
        return sum(self._gameRewards) / len(self._gameRewards)

    def get_reward_list(self):
        """ returns all the rewards from all game to later be used for creation of histogram """
        return self._gameRewards

    def get_loss_list(self):
        return self._loss

    def get_max(self):
        """ returns maximum reward"""
        return max(self._gameRewards)

    def get_min(self):
        """ returns minimum reward"""
        return min(self._gameRewards)

    def get_probability_loss(self):
        """ returns the probability of a loss """
        count_loss = 0
        for value in self._gameRewards:
            if value < 0:
                count_loss += 1
        return count_loss / len(self._gameRewards)

    def get_ave_probability_loss(self):
        """ returns the probability of a loss """
        count_loss = 0
        for value in self._gameRewards:
            if value < 0:
                count_loss += 1
        return sum(self._loss) / len(self._gameRewards)



class CohortOutcomes:
    def __init__(self,simulated_cohort):
        self._simulatedCohort = simulated_cohort

        self._sumStat_rewards = \
            Stat.SummaryStat('The total rewards', self._simulatedCohort.get_reward_list())
        self._sumStat_loss = \
            Stat.SummaryStat('The probability of loss', self._simulatedCohort.get_loss_list())
    def get_ave_rewards(self):
        return self._sumStat_rewards.get_mean()

    def get_CI_expected_rewards(self,alpha):
        return self._sumStat_rewards.get_t_CI(alpha)

    def get_CI_probability_loss(self,alpha):
        return self._sumStat_loss.get_t_CI(alpha)

class Multicohort:
    def __init__(self, ids, n_games, prob_heads):
        self._ids = ids
        self._n_games = n_games
        self._probheads=prob_heads

        self._ExpectedRewards = []
        self._mean_ExpectedRewards=[]
        self._Uncertainty = []
        self._mean_uncertainty = []
        self._sumStat_meanRewards = None
        self._sumStat_uncertainty= None

    def simulatee(self):
        for i in range(len(self._ids)):
            game = SetOfGames(self._probheads[i],self._n_games[i])
            output=game.simulation()
            self._ExpectedRewards.append(game.get_reward_list())
            self._Uncertainty.append(game.get_loss_list())
            self._mean_ExpectedRewards.append(game.get_ave_reward())
        self._sumStat_meanRewards = Stat.SummaryStat("Mean rewards", self._mean_ExpectedRewards)
        self._sumStat_uncertainty = Stat.SummaryStat("Mean loss", self._mean_uncertainty)

###############ExpectedRewards

    def get_all_mean_ExpectedRewards(self):
        return self._mean_ExpectedRewards

    def get_overall_mean_ExpectedRewards(self):
        return self._sumStat_meanRewards.get_mean()

    def get_cohort_PI_ExpectedRewards(self, cohort_index, alpha):

        st = Stat.SummaryStat('', self._ExpectedRewards[cohort_index])
        return st.get_PI(alpha)

    def get_PI_mean_ExpectedRewards(self, alpha):
        """ :returns: the prediction interval of the mean survival time"""
        return self._sumStat_meanRewards.get_PI(alpha)

    def get_overall_mean_rewards(self):
        return self._mean_ExpectedRewards.get_mean()

    def get_cohort_PI_ExpectedRewards(self,cohort_index,alpha):
        st = Stat.SummaryStat('', self._ExpectedRewards[cohort_index])
        return st.get_PI(alpha)

    def get_cohort_PI_mean_ExpectedRewards(self,alpha):
        return self._sumStat_meanRewards.get_PI(alpha)

###############lossprob

    def get_all_mean_Uncertainty(self):
        return self._mean_uncertainty

    def get_overall_mean_Uncertainty(self):
        return self._sumStat_uncertainty.get_mean()

    def get_cohort_PI_Uncertainty(self, cohort_index, alpha):
        st = Stat.SummaryStat('', self._Uncertainty[cohort_index])
        return st.get_PI(alpha)

    def get_PI_mean_Uncertainty(self, alpha):
        """ :returns: the prediction interval of the mean survival time"""
        return self._sumStat_uncertainty.get_PI(alpha)

    def get_overall_mean_Uncertainty(self):
        return self._mean_uncertainty.get_mean()

    def get_cohort_PI_Uncertainty(self, cohort_index, alpha):
        st = Stat.SummaryStat('', self._Uncertainty[cohort_index])
        return st.get_PI(alpha)

    def get_cohort_PI_mean_Uncertainty(self, alpha):
        return self._sumStat_uncertainty.get_PI(alpha)


# Problem 1: Confidence Interval

# Calculate expected reward of 1000 games
prob_head=0.5
n_games=1000
alpha=0.05
TIME_STEPS=20
NUM_SIM_COHORTS = 10

trial = SetOfGames(prob_head=prob_head, n_games=n_games)
print("The average expected reward is:", trial.get_ave_reward())
print("The average probability of the expected loss is:", trial.get_ave_probability_loss())


cohortOutcome=trial.simulation()
print("The 95% t-based CI for the expected reward:", cohortOutcome.get_CI_expected_rewards(alpha=alpha))
print("The 95% t-based CI for the expected probability of loss:", cohortOutcome.get_CI_probability_loss(alpha=alpha))

#If we play the games for 1000 times,
# we get 95 times of the confidence interval for the expected reward[-31.79, -20.01]
# that covered the true mean (-25.9).

# we get 95 times of the confidenc interval for the expected probability of loss[6.26%,11.94%]
# that covered the true mean (9.10%)




gambler=Multicohort(ids=range(n_games), n_games=[n_games]*n_games, prob_heads=[prob_head]*n_games)
gambler.simulatee()
print('Projected mean expected rewards',
      gambler.get_overall_mean_Uncertainty())
# print projection interval
print('95% projection interval of ExpectedRewards',
      gambler.get_cohort_PI_mean_Uncertainty(alpha))
print('Projected mean uncertainty',
      gambler.get_overall_mean_Uncertainty())
# print projection interval
print('95% projection interval of uncertainty',
      gambler.get_cohort_PI_mean_Uncertainty(alpha))



owner=Multicohort(ids=range(n_games), n_games=[n_games]*NUM_SIM_COHORTS, prob_heads=[prob_head]*NUM_SIM_COHORTS)
owner.simulatee()
print('Projected mean expected rewards',
      owner.get_all_mean_ExpectedRewards())
# print projection interval
print('95% projection interval of ExpectedRewards',
      owner.get_cohort_PI_ExpectedRewards(alpha))
print('Projected mean uncertainty',
      owner.get_overall_mean_Uncertainty())
# print projection interval
print('95% projection interval of uncertainty',
      owner.get_cohort_PI_mean_Uncertainty(alpha))