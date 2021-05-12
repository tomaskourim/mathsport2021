-- 12 372 zapasu
select *
from matches
where start_time_utc >= '2021-02-01 00:00:00.000000' and start_time_utc < '2021-05-01 00:00:00.000000';

-- 3 642 zapasu
select distinct match_id
from match_course
         join matches m on m.id = match_course.match_id
where start_time_utc >= '2021-02-01 00:00:00.000000' and start_time_utc < '2021-05-01 00:00:00.000000';

-- kompletni zapasy (Grand Slam zvlast)

-- zapasy, kde mam alespon 2 sety (i.e. jedna teoreticka sazka)

-- sety, na ktere lze sazet
