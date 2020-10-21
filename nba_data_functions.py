from tqdm import tqdm
import numpy as np
import pandas as pd

# For getting main df
def get_main_data():
    '''Returns a df with starting data for NBA players from 1999-2000 season through 2015-16 season. 
    In the process unpickles a dict of city data, a dict of NBA player stats, and a df of All-Star data.
    Arguments: (none)
    Returns: df
    '''

    # Unpickling
    with open("full_team_dict.pickle", "rb") as to_read:
        full_team_dict = pickle.load(to_read)
    with open("all_star_df.pickle", "rb") as to_read:
        all_stars2000_2016 = pickle.load(to_write)
    with open("master_df.pickle", "rb") as to_read:
        master_df = pickle.load(to_write)

    # Reading csv in as a df and making a few quick changes
    master_df = master_df[(master_df["Year"] >= 2000.0) & (master_df["Year"] <= 2016.00) & (master_df["Year"] != 1999.0)].drop(columns="Unnamed: 0")
    master_df["Year"] = master_df["Year"].astype(int)
    master_df["Player"] = master_df["Player"].str.strip("*")
    master_df.drop(columns=["blanl", "blank2"], inplace=True)
    master_df = master_df.reset_index()
    master_df.drop(columns="index", inplace=True)

    # Adding All-Star info to existing player df
    all_star_yes_no_column = []
    all_star_name_column = []
    all_star_how_voted_column = []
    for year in master_df["Year"].unique():
        all_players_df = master_df[master_df["Year"] == year]
        year_players = all_players_df["Player"]
        all_stars_df = all_stars2000_2016[all_stars2000_2016["Year"] == year]
        for guy in year_players:
            all_star_name_column.append(guy)
            if guy == "Metta World Peace" or guy == "Ron Artest":
                if "Metta World Peace" in list(all_stars_df["Player"]) or "Ron Artest" in list(all_stars_df["Player"]):
                    all_star_yes_no_column.append("Yes")
                    selection_type = all_stars_df[all_stars_df["Player"] == guy]["Selection Type"]
                    all_star_how_voted_column.append(selection_type)
                else:
                    all_star_yes_no_column.append("No")
                    all_star_how_voted_column.append("N/a")
            else:
                if guy in list(all_stars_df["Player"]):
                    all_star_yes_no_column.append("Yes")
                    selection_type = all_stars_df[all_stars_df["Player"] == guy]["Selection Type"]
                    all_star_how_voted_column.append(selection_type)
                else:
                    all_star_yes_no_column.append("No")
                    all_star_how_voted_column.append("N/a")

    all_star_details_df = pd.DataFrame({"Player (check)": all_star_name_column, "All-Star?": all_star_yes_no_column, "Selection process": all_star_how_voted_column})

    # Updating resulting df and then adding it to "master_df"
    master_df[all_star_details_df.columns] = all_star_details_df

    # Removing duplicate columns for guys in a season in which they got traded (teams + "TOT"; keeping "TOT" with some tweaks)
    trade_dupes = []
    years = list(master_df["Year"].unique())
    for year in years:
        all_players_df = master_df[master_df["Year"] == year]
        year_players = all_players_df["Player"]
        players_list = list(master_df[master_df["Year"] == year]["Player"])
        for guy in list(set(players_list)):
            if players_list.count(guy) > 1:
                if len(master_df[(master_df["Player"] == guy) & (master_df["Tm"] != "TOT") & (master_df["Year"] == year)]["Age"].unique()) == 1:
                    dupe_players_index = master_df[(master_df["Player"] == guy) & (master_df["Tm"] != "TOT") & (master_df["Year"] == year)].sort_values(by="G", ascending=False).index
                    new_team_index = dupe_players_index[0]
                    trade_dupes.extend(list(dupe_players_index))
                    tot_index = master_df[(master_df["Player"] == guy) & (master_df["Tm"] == "TOT") & (master_df["Year"] == year)].index[0]
                    new_team = master_df.iloc[new_team_index]["Tm"]
                    master_df.iloc[tot_index, 4] = new_team

    # Dropping last remaining "TOT" set for Marcus Williams in 2008
    marcus_williams_dupe_index = master_df[(master_df["Player"] == "Marcus Williams") & (master_df["Tm"] != "TOT") & (master_df["Year"] == 2008) & (master_df["Age"] == 21.0)].sort_values(by="G", ascending=False).index
    new_marcus_team_index = marcus_williams_dupe_index[0]
    trade_dupes.extend(list(marcus_williams_dupe_index))
    tot_marcus_index = master_df[(master_df["Player"] == "Marcus Williams") & (master_df["Tm"] == "TOT") & (master_df["Year"] == 2008) & (master_df["Age"] == 21.0)].index[0]
    new_team = master_df.iloc[new_marcus_team_index]["Tm"]
    master_df.iloc[tot_marcus_index, 4] = new_team

    # Dropping the duplicate rows 
    trade_dupes.sort(reverse=True)
    for row in trade_dupes:
        master_df.drop(row, inplace=True)

    # Adding a column called "All-Star next season?"
    sorted_master_df = master_df.sort_values(by=["Player", "Year"]).reset_index().drop(columns=["index", "Player (check)"])
    next_year_column = []
    for i in range(0, (len(list(sorted_master_df["Player"])))-1):
        if sorted_master_df.iloc[i]["Player"] == sorted_master_df.iloc[(i+1)]["Player"]:
            if sorted_master_df.iloc[(i+1)]["All-Star?"] == "Yes":
                next_year_column.append("Yes")
            else:
                next_year_column.append("No")
        else:
            next_year_column.append("No")

    # Because last player in his last year (like Big Z) can't make All-Star Game next year
    next_year_column.append("No")
    sorted_master_df["All-Star next season?"] = next_year_column

    # Converting "All-Star" and "All-Star next year?" columns to 1s and 0s
    sorted_master_df["All-Star next season?"] = sorted_master_df["All-Star next season?"].replace(to_replace={"Yes": 1, "No": 0})
    sorted_master_df["All-Star?"] = sorted_master_df["All-Star?"].replace(to_replace={"Yes": 1, "No": 0})

    # Creating new column and re-indexing for ease of recognition
    sorted_master_df["Player, year"] = sorted_master_df["Player"] + ', ' + sorted_master_df["Year"].astype(str)
    reindexed_master_df = sorted_master_df.set_index("Player, year")

    # Fixing BRK to be BKN and PHX to be PHO (in "Tm")
    reindexed_master_df["Tm"].replace(to_replace={"BRK": "BKN", "PHO": "PHX"}, inplace=True)

    # Adding "TV market size" column
    reindexed_master_df["TV market size"] = reindexed_master_df["Tm"].apply(lambda x: full_team_dict[x]["2000-01 TV market size"])

    # 3 rows had NaNs in PER column (played < 5 games each)
    reindexed_master_df = reindexed_master_df.drop(reindexed_master_df[reindexed_master_df["PER"].isna()].index)

    # Dropping guys < 5 games
    reindexed_master_df = reindexed_master_df.drop(reindexed_master_df[(reindexed_master_df["G"] < 5) & (reindexed_master_df["Player"] != "Jeremy Lin")].index)

    # Filling TOR and VAN "TV market size" data with mean
    market_mean = reindexed_master_df["TV market size"].mean()
    reindexed_master_df["TV market size"].fillna(market_mean, inplace=True)

    # Dropping 2015-16 season (to be used later as holdout season 1)
    reindexed_master_df = reindexed_master_df.drop(reindexed_master_df[reindexed_master_df["Year"] == 2016].index)

    return reindexed_master_df


# For adding contextualizing features for counting stats
def per_game_rel_to_season(df, list_of_features):
    '''Takes a df and a list of included columns and returns the same df with three new columns.
    Columns must be stat of type int or float. The new columns will host the same stat, but:
    -[stat]/per game
    -[stat]/per game (season mean for all players)
    -[stat]/per game relative to season mean for all players

    Arguments: df, list of columns in df of type int or float
    Returns: df
    '''

    for feature in tqdm(list_of_features):
        per_game_feature = f'{feature}/game'
        per_game_feature_list = []
        for row in tqdm(range(df.shape[0])):
            per_game_feature_list.append(round(df.iloc[row][feature] / df.iloc[row]["G"], 1))
        df[per_game_feature] = per_game_feature_list

        relative_feature = f'{feature}/game relative'
        season_mean_feature = f'{feature}/game average'
        relative_feature_list = []
        season_mean_list = []
        for row in range(df.shape[0]):
            season = df.iloc[row]["Year"]
            season_mean = df[df["Year"] == season][per_game_feature].mean()
            season_mean_list.append(round(season_mean, 1))
            season_std = df[df["Year"] == season][per_game_feature].std()
            relative_feature_list.append(round((df.iloc[row][per_game_feature] - season_mean)/season_std, 1))
        df[season_mean_feature] = season_mean_list
        df[relative_feature] = relative_feature_list

    return df