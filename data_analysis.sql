SET myvars.start_date TO '2021-02-01 00:00:00.000000';
SET myvars.end_date TO '2021-05-01 00:00:00.000000';


-- 12 372 matches total
SELECT *
FROM matches
WHERE start_time_utc >= CURRENT_SETTING('myvars.start_date')::timestamptz AND
    start_time_utc < CURRENT_SETTING('myvars.end_date')::timestamptz;

-- 3 642 matches with at least one set result
SELECT DISTINCT match_id
FROM match_course
         JOIN matches m ON m.id = match_course.match_id
WHERE start_time_utc >= CURRENT_SETTING('myvars.start_date')::timestamptz AND
    start_time_utc < CURRENT_SETTING('myvars.end_date')::timestamptz;

-- 3 365 matches with complete results
SELECT *
FROM (
    SELECT match_id, MAX(sets_won) AS winning_sets
    FROM (
        SELECT match_id, COUNT(*) AS sets_won
        FROM match_course
        WHERE utc_time_recorded >= CURRENT_SETTING('myvars.start_date')::timestamptz AND
            utc_time_recorded < CURRENT_SETTING('myvars.end_date')::timestamptz
        GROUP BY match_id, result
    ) AS sets_won
    GROUP BY match_id
) AS winning_results
         JOIN matches ON winning_results.match_id = matches.id
         JOIN tournament t ON matches.tournament_id = t.id
WHERE winning_sets = 2
        AND NOT (name IN ('ATP Australian Open', 'ATP US Open', 'ATP Wimbledon', 'ATP French Open') AND sex = 'men') OR
    winning_sets = 3
            AND (name IN ('ATP Australian Open', 'ATP US Open', 'ATP Wimbledon', 'ATP French Open') AND sex = 'men')
ORDER BY match_id;


-- 3 438 matches with at least 2 sets played and recorded (i.e. at least 1 possible bet)
SELECT *
FROM (
    SELECT match_id, COUNT(*) AS sets_played
    FROM match_course
    WHERE utc_time_recorded >= CURRENT_SETTING('myvars.start_date')::timestamptz AND
        utc_time_recorded < CURRENT_SETTING('myvars.end_date')::timestamptz
    GROUP BY match_id) AS all_sets_played
WHERE sets_played >= 2
ORDER BY match_id;


-- 8 310 sets with complete information (TODO maybe there are some matches with odds for set1 and set3, but not set2)
SELECT home,
    away,
    set_number,
    odd1,
    odd2,
    result,
    start_time_utc
FROM (
    SELECT *,
        CASE
            WHEN match_part = 'set1'
                THEN 1
            WHEN match_part = 'set2'
                THEN 2
            WHEN match_part = 'set3'
                THEN 3
            WHEN match_part = 'set4'
                THEN 4
            WHEN match_part = 'set5'
                THEN 5
            END AS set_number_odds
    FROM odds) AS odds_enhanced
         INNER JOIN
(SELECT *, ma.id AS matchid
 FROM matches_bookmaker mb
          JOIN matches ma ON mb.match_id = ma.id
          JOIN match_course mc ON mb.match_id = mc.match_id) AS match_course_enhanced
ON odds_enhanced.match_bookmaker_id = match_course_enhanced.match_bookmaker_id AND
    odds_enhanced.bookmaker_id = match_course_enhanced.bookmaker_id AND
    odds_enhanced.set_number_odds = match_course_enhanced.set_number
WHERE start_time_utc >= CURRENT_SETTING('myvars.start_date')::timestamptz AND
    start_time_utc < CURRENT_SETTING('myvars.end_date')::timestamptz
ORDER BY match_course_enhanced.matchid, match_course_enhanced.set_number;


-- 3 642 first sets with complete information
SELECT home,
    away,
    set_number,
    odd1,
    odd2,
    case when result = 'home' then 1 else 0 end as result,
    start_time_utc
FROM (
    SELECT *,
        CASE
            WHEN match_part = 'set1'
                THEN 1
            WHEN match_part = 'set2'
                THEN 2
            WHEN match_part = 'set3'
                THEN 3
            WHEN match_part = 'set4'
                THEN 4
            WHEN match_part = 'set5'
                THEN 5
            END AS set_number_odds
    FROM odds) AS odds_enhanced
         INNER JOIN
(SELECT *, ma.id AS matchid
 FROM matches_bookmaker mb
          JOIN matches ma ON mb.match_id = ma.id
          JOIN match_course mc ON mb.match_id = mc.match_id
          join tournament t on ma.tournament_id = t.id) AS match_course_enhanced
ON odds_enhanced.match_bookmaker_id = match_course_enhanced.match_bookmaker_id AND
    odds_enhanced.bookmaker_id = match_course_enhanced.bookmaker_id AND
    odds_enhanced.set_number_odds = match_course_enhanced.set_number
WHERE start_time_utc >= CURRENT_SETTING('myvars.start_date')::timestamptz AND
    start_time_utc < CURRENT_SETTING('myvars.end_date')::timestamptz and set_number = 1
ORDER BY match_course_enhanced.matchid, match_course_enhanced.set_number;
