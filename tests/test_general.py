import unittest

import numpy as np

from optimal_fair_odds_parameter import get_fair_odds


class GeneralTest(unittest.TestCase):

    def test_fair_odds(self):
        odds = np.array([4.42, 1.17])
        parameters = [0, 2, 0.7639320225002102, 0.5]
        probability_pairs = [[0.16224116360915253, 0.8377588363908474], [0.20930232558139533, 0.7906976744186046],
                             [0.18021692793248525, 0.8197830720675147], [0.17400645410221324, 0.8259935458977867]]
        for index, parameter in enumerate(parameters):
            computed_probabilities = get_fair_odds(odds, parameter)
            self.assertEqual(computed_probabilities[0], probability_pairs[index][0])
            self.assertEqual(computed_probabilities[1], probability_pairs[index][1])

        odds = np.array([2.65, 1.42])
        parameters = [0]
        probability_pairs = [[0.32423878955901275, 0.6757612104409872]]
        for index, parameter in enumerate(parameters):
            computed_probabilities = get_fair_odds(odds, parameter)
            self.assertEqual(computed_probabilities[0], probability_pairs[index][0])
            self.assertEqual(computed_probabilities[1], probability_pairs[index][1])


if __name__ == '__main__':
    unittest.main()
