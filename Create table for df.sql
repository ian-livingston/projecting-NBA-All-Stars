\connect ian;

CREATE TABLE project_3 (season NUMERIC, player VARCHAR, pos VARCHAR, age NUMERIC, tm VARCHAR, g NUMERIC, gs NUMERIC, mp NUMERIC, per NUMERIC, \
ts_percentage NUMERIC, threes_par NUMERIC, ftr NUMERIC, orb_percentage NUMERIC, drb_percentage NUMERIC, trb_percentage NUMERIC, ast_percentage NUMERIC, \
stl_percentage NUMERIC, blk_percentage NUMERIC, tov_percentage NUMERIC, usg_percentage NUMERIC, ows NUMERIC, dws NUMERIC, ws NUMERIC, \
ws_per_forty_eight_minutes NUMERIC, obpm NUMERIC, dbpm NUMERIC, bpm NUMERIC, vorp NUMERIC, fg NUMERIC, fga NUMERIC, fg_percentage NUMERIC, threes NUMERIC, \
threes_attempted NUMERIC, three_point_percentage NUMERIC, twos NUMERIC, twos_attempted NUMERIC, two_point_percentage NUMERIC, efg_percentage NUMERIC, ft NUMERIC, \
fta NUMERIC, ft_percentage NUMERIC, orb NUMERIC, drb NUMERIC, trb NUMERIC, ast NUMERIC, stl NUMERIC, blk NUMERIC, tov NUMERIC, pf NUMERIC, pts NUMERIC, \
all_star NUMERIC, selection_process VARCHAR, all_star_next_season NUMERIC, tv_market NUMERIC), AST/game NUMERIC, AST/game_average NUMERIC, AST/game_relative NUMERIC, \
PTS/game NUMERIC, PTS/game_average NUMERIC, PTS/game_relative NUMERIC, ORB/game NUMERIC, ORB/game_average NUMERIC, ORB/game_relative NUMERIC, DRB/game NUMERIC, \
DRB/game_average NUMERIC, DRB/game relative NUMERIC, TRB/game NUMERIC, TRB/game_average NUMERIC, TRB/game_relative NUMERIC, STL/game NUMERIC, STL/game_average, \
NUMERIC, STL/game_relative NUMERIC, BLK/game NUMERIC, BLK/game_average NUMERIC, BLK/game_relative NUMERIC, MP/game NUMERIC, MP/game_average NUMERIC, MP/game relative NUMERIC, \
3P/game NUMERIC, 3P/game_average NUMERIC, 3P/game_relative NUMERIC, FT/game NUMERIC, FT/game_average NUMERIC, FT/game_relative NUMERIC, FTA/game NUMERIC, FTA/game_average NUMERIC, \
FTA/game_relative NUMERIC);

COPY project_3
FROM '/Users/ian/METIS bootcamp/Project 3/Main df.csv' DELIMITER ',' CSV HEADER;