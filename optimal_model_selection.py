from typing import List, Tuple

import numpy as np
import pandas as pd

from database_operations import execute_sql_postgres
from odds_to_probabilities import get_fair_odds_parameter, get_fair_odds
from utils import get_logger, COLUMN_NAMES


def get_matches_data(start_date: str, end_date: str) -> pd.DataFrame:
    query = "SELECT matchid, home,     away,     set_number,     odd1,     odd2,     " \
            "case when result = 'home' then 1 else 0 end as result,     start_time_utc FROM (     " \
            "SELECT *,         CASE             WHEN match_part = 'set1'                 THEN 1             " \
            "WHEN match_part = 'set2'                 THEN 2             " \
            "WHEN match_part = 'set3'                 THEN 3             " \
            "WHEN match_part = 'set4'                 THEN 4             " \
            "WHEN match_part = 'set5'                 THEN 5             " \
            "END AS set_number_odds     FROM odds) AS odds_enhanced          " \
            "INNER JOIN " \
            "(SELECT *, ma.id AS matchid  FROM matches_bookmaker mb           " \
            "JOIN matches ma ON mb.match_id = ma.id           " \
            "JOIN match_course mc ON mb.match_id = mc.match_id) AS match_course_enhanced " \
            "ON odds_enhanced.match_bookmaker_id = match_course_enhanced.match_bookmaker_id " \
            "AND     odds_enhanced.bookmaker_id = match_course_enhanced.bookmaker_id " \
            "AND     odds_enhanced.set_number_odds = match_course_enhanced.set_number" \
            " WHERE start_time_utc >= %s AND     " \
            "start_time_utc < %s " \
            "ORDER BY start_time_utc, match_course_enhanced.matchid, match_course_enhanced.set_number"
    return pd.DataFrame(execute_sql_postgres(query, [start_date, end_date], False, True), columns=COLUMN_NAMES)


def transform_data(matches_data: pd.DataFrame) -> Tuple[List[List[int]], List[float]]:
    # transform walks into List[List[int]
    # get p0 for each walk

    fair_odds_parameter = get_fair_odds_parameter()

    walks = []
    starting_probabilities = []
    matchid = ""
    walk = []
    for index, set_data in matches_data.iterrows():
        if matchid != set_data.matchid:
            matchid = set_data.matchid
            odds = np.array([set_data.odd1, set_data.odd2])
            probabilities = get_fair_odds(odds, fair_odds_parameter)
            starting_probabilities.append(probabilities[0])
            walks.append(walk)
            walk = [set_data.result]
        else:
            walk.append(set_data.result)
    walks.append(walk)
    walks.pop()

    for index, walk in enumerate(walks):
        if len(walk) == 1:
            del walks[index]
            del starting_probabilities[index]

    return walks, starting_probabilities


def main():
    # get all data
    start_date = '2021-02-01 00:00:00.000000'
    end_date = '2021-05-01 00:00:00.000000'
    matches_data = get_matches_data(start_date, end_date)
    # transform data
    walks, starting_probabilities = transform_data(matches_data)
    # get model estimate + parameters
    # repeat for subgroups
    logger.info("Done")


if __name__ == '__main__':
    logger = get_logger()
    main()
