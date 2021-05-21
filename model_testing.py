from optimal_model_selection import get_matches_data, transform_data, get_optimal_model
from utils import get_logger


def test_model(walks, starting_probabilities, all_matches_set_odds, c_lambdas, model_type):
    for walk,starting_probability,single_match_set_odds in zip(walks, starting_probabilities, all_matches_set_odds):
        print()


def main():
    # get all matches from testing dataset
    start_date = '2021-05-01 00:00:00.000000'
    end_date = '2021-05-20 00:00:00.000000'
    matches_data = get_matches_data(start_date, end_date)

    # transform data
    walks, starting_probabilities, all_matches_set_odds = transform_data(matches_data)

    # compute probability and odds for each set
    c_lambdas, model_type = get_optimal_model()

    # compare with real odds
    test_model(walks, starting_probabilities, all_matches_set_odds, c_lambdas, model_type)


if __name__ == '__main__':
    logger = get_logger()
    main()
