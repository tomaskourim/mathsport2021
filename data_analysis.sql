SET myvars.start_date TO '2021-02-01 00:00:00.000000';
SET myvars.end_date TO '2021-05-01 00:00:00.000000';


-- 12 372 matches total
SELECT *
FROM matches
WHERE start_time_utc >= CURRENT_SETTING('myvars.start_date')::timestamptz
  AND start_time_utc < CURRENT_SETTING('myvars.end_date')::timestamptz;

-- 3 642 matches with at least one set result
SELECT DISTINCT match_id
FROM match_course
         JOIN matches m ON m.id = match_course.match_id
WHERE start_time_utc >= CURRENT_SETTING('myvars.start_date')::timestamptz
  AND start_time_utc < CURRENT_SETTING('myvars.end_date')::timestamptz;

-- 3 365 matches with complete results
SELECT *
FROM (
         SELECT match_id, MAX(sets_won) AS winning_sets
         FROM (
                  SELECT match_id, COUNT(*) AS sets_won
                  FROM match_course
                  WHERE utc_time_recorded >= CURRENT_SETTING('myvars.start_date')::timestamptz
                    AND utc_time_recorded < CURRENT_SETTING('myvars.end_date')::timestamptz
                  GROUP BY match_id, result
              ) AS sets_won
         GROUP BY match_id
     ) AS winning_results
         JOIN matches ON winning_results.match_id = matches.id
         JOIN tournament t ON matches.tournament_id = t.id
WHERE winning_sets = 2
    AND NOT (name IN ('ATP Australian Open', 'ATP US Open', 'ATP Wimbledon', 'ATP French Open') AND sex = 'men')
   OR winning_sets = 3
    AND (name IN ('ATP Australian Open', 'ATP US Open', 'ATP Wimbledon', 'ATP French Open') AND sex = 'men')
ORDER BY match_id;


-- zapasy, kde mam alespon 2 sety (i.e. jedna teoreticka sazka)

-- sety, na ktere lze sazet
