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


# For creating a column with values quantifying a guy's TV market value that year
# This version does not account for guy's who are not rookies in 2000 (first year of data)
def tv_market_cumulative(ordered_df):

    profile_column = []
    for i in tqdm(range(ordered_df.shape[0])):
        if i == 0:
            total_profile = ordered_df.iloc[0]["TV market size"]
        else:
            x = i
            total_market = []
            current_guy = ordered_df.iloc[x]["Player"]
            this_year_market = ordered_df.iloc[x]["TV market size"]
            total_market.append(this_year_market)
            previous_guy = ordered_df.iloc[x-1]["Player"]
            if current_guy == previous_guy:
                same_guy = True
                while same_guy == True:
                    last_year_market = ordered_df.iloc[x-1]["TV market size"]
                    total_market.append(last_year_market)
                    x -= 1
                    same_guy = ordered_df.iloc[x]["Player"] == reindexed_master_df.sort_values(by=["Player", "Year"]).iloc[x-1]["Player"]
                for _ in range(0, len(total_market[::-1])):
                    if _ == 0:
                        profile = total_market[::-1][0] * 0.75     # Rookies get 0.75 of full value
                        best_profile_year = 0
                    else:
                        if total_market[::-1][_] > total_market[::-1][_-1]:
                            if _ > 1:
                                exponent = _ - 1
                                profile += total_market[::-1][_] * (1.1**exponent)
                            else:
                                profile += total_market[::-1][_]
                            best_profile_year = _
                        else:
                            if _ > 1:
                                exponent = _ - 1
                                profile += total_market[::-1][best_profile_year] * (1.1**exponent)
                            else:
                                profile += total_market[::-1][best_profile_year]
            else:
                profile = total_market[0] * 0.75      # Rookies get 0.75 of full value
            total_profile = profile/len(total_market)
        profile_column.append(round(total_profile, 0))

    ordered_df["Adjusted TV market value"] = profile_column
    
    return ordered_df


# For creating a column with values quantifying a guy's projected trajectory
def get_trajectory(ordered_df):

    projected_changes = []
    for i in tqdm(range(1, ordered_df.shape[0])):
        x = i
        current_guy = reindexed_master_df.sort_values(by=["Player", "Year"]).iloc[x]["Player"]
        previous_guy = reindexed_master_df.sort_values(by=["Player", "Year"]).iloc[x-1]["Player"]
        if current_guy == previous_guy:
            same_guy = True
            change_vector = []
            while same_guy == True:
                this_year_stat = reindexed_master_df.sort_values(by=["Player", "Year"]).iloc[x]["PTS/game"]
                last_year_stat = reindexed_master_df.sort_values(by=["Player", "Year"]).iloc[x-1]["PTS/game"]
                stat_change = this_year_stat - last_year_stat
                change_vector.append(stat_change)
                x -= 1
                same_guy = reindexed_master_df.sort_values(by=["Player", "Year"]).iloc[x]["Player"] == reindexed_master_df.sort_values(by=["Player", "Year"]).iloc[x-1]["Player"]
            if len(change_vector) > 1:
                change_change_vector = [(change_vector[_] - change_vector[_+1]) for _ in range(0, (len(change_vector)-1))]
                if len(change_change_vector) > 1:
                    bump = (change_change_vector[0] + change_change_vector[1])/2
                    if change_change_vector[0] > 5:
                        bump *= 1.25
                    elif change_change_vector[0] < -5:
                        bump -= 1.25
                elif len(change_change_vector) == 1:
                    bump = change_change_vector[0]
                projected_changes.append(bump)
            else:
                projected_changes.append(stat_change)
        else:
            projected_changes.append(0)

    ordered_df["Trajectory"] = projected_changes
    
    return ordered_df


# For getting clutch stats from stats.NBA.com
def get_clutch_stats(list_of_seasons):

    chromedriver = "/Applications/chromedriver" # path to the chromedriver executable
    os.environ["webdriver.chrome.driver"] = chromedrive
    clutch_dict = {}
    for x, season in tqdm(enumerate(list_of_seasons)):
        driver = webdriver.Chrome(chromedriver)
        driver.get("https://stats.nba.com/players/clutch-traditional/?sort=PTS&dir=-1&Season={}&SeasonType=Regular%20Season&PerMode=Totals".format(season))
        time.sleep(10)
        soup = BeautifulSoup(driver.page_source)
        num_clicks_needed = len(soup.find("div", class_="stats-table-pagination__info").find_all("option")) - 1
        year = '20' + str(season)[-2:]
        year_int = int(year)
        if x == 0:
            clutch_minutes_list = []
            clutch_points_list = []
            clutch_assists_list = []
            clutch_fantasty_points_list = []
            names_list = []
            years_list = []
        
        for i in tqdm(range(num_clicks_needed)):
            for player in soup.find("tbody").find_all("tr"):
                years_list.append(year_int)
                name = player.find_all("td")[1].text.strip()
                names_list.append(name)
                clutch_minutes = player.find_all("td")[7].text
                clutch_minutes_list.append(float(clutch_minutes))
                clutch_points = player.find_all("td")[8].text
                clutch_points_list.append(float(clutch_points))
                clutch_assists = player.find_all("td")[21].text
                clutch_assists_list.append(float(clutch_assists))
                clutch_fantasty_points = player.find_all("td")[26].text
                clutch_fantasty_points_list.append(float(clutch_fantasty_points))
            next_page = driver.find_element_by_xpath('//a[@class="stats-table-pagination__next"]')
            next_page.click()
            soup = BeautifulSoup(driver.page_source)

        print(len(clutch_minutes_list), len(clutch_points_list), len(clutch_assists_list), len(clutch_fantasty_points_list), len(years_list))
        time.sleep(5)

    clutch_dict["Player"] = names_list
    clutch_dict["Year"] = years_list
    clutch_dict["Clutch minutes"] = clutch_minutes_list
    clutch_dict["Clutch points"] = clutch_points_list
    clutch_dict["Clutch assists"] = clutch_assists_list
    clutch_dict["Clutch fantasy points"] = clutch_fantasty_points_list
    
    clutch_stats_df = pd.DataFrame.from_dict(clutch_dict)

    with open("clutch.pickle", "wb") as to_write:
        pickle.dump(clutch_stats_df, to_write)

    return clutch_stats_df


# For making a column with values reflecting whether a guy has previously made 0, 1-3, 4-7, or 8+ All-Star Games
# This currently cannot account for players who made All-Star Games before 2000-01 season
# First function is for binning returned years
def all_star_history_bins(single_year_value):
    if single_year_value >= 8:
        all_star_bin = "8+"
    elif single_year_value >= 4:
        all_star_bin = "4-7"
    elif single_year_value >= 1:
        all_star_bin = "1-3"
    else:
        all_star_bin = "0"
    
    return all_star_bin


def get_past_all_star_games(ordered_df):

    all_star_history = []
    for i in tqdm(range(ordered_df.shape[0])):
        x = i
        current_guy = ordered_df.iloc[x]["Player"]
        previous_guy = ordered_df.iloc[x-1]["Player"]
        if i == 0:
            all_star_selections_to_date = 0
            if ordered_df.iloc[x]["All-Star?"] == 1:
                all_star_selections_to_date += 1
            all_star_history.append(all_star_selections_to_date)
            print(current_guy, all_star_selections_to_date)
        else:
            if current_guy == previous_guy:
                if ordered_df.iloc[x]["All-Star?"] == 1:
                    all_star_selections_to_date += 1
            else:
                all_star_selections_to_date = 0
                if ordered_df.iloc[x]["All-Star?"] == 1:
                    all_star_selections_to_date += 1
            all_star_history.append(all_star_selections_to_date)
            print(current_guy, all_star_selections_to_date)

    ordered_df["Past All-Star Games (incl this season)"] = all_star_history
    ordered_df["Past All-Star Games (incl this season)"] = ordered_df["Past All-Star Games (incl this season)"].apply(lambda x: all_star_history_bins(x))
    
    return ordered_df


# For getting links to all NBA player photos
def get_photo_links(list_of_seasons):

    import pandas as pd
    import numpy as np
    import time
    import requests
    import re
    from bs4 import BeautifulSoup
    from fake_useragent import UserAgent
    import pickle
    from tqdm import tqdm
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    import time, os

ua = UserAgent()
user_agent = {'User-agent': ua.random}
chromedriver = "/Applications/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver

    photo_lists = []
    for x, season in tqdm(enumerate(list_of_seasons)):
        driver = webdriver.Chrome(chromedriver)
        driver.get("https://stats.nba.com/players/clutch-traditional/?sort=PTS&dir=-1&Season={}&SeasonType=Regular%20Season&PerMode=Totals".format(season))
        time.sleep(10)
        soup = BeautifulSoup(driver.page_source)
        num_clicks_needed = len(soup.find("div", class_="stats-table-pagination__info").find_all("option")) - 1
        year = '20' + str(season)[-2:]
        year_int = int(year)
        
        for i in tqdm(range(num_clicks_needed)):
            for player in soup.find("tbody").find_all("tr"):
                years_list.append(year_int)
                name = player.find_all("td")[1].text.strip()
                photo_link = player.find_all("td")[1]["href"]
                photo_lists.append(name, photo_link)   
            next_page = driver.find_element_by_xpath('//a[@class="stats-table-pagination__next"]')
            next_page.click()
            soup = BeautifulSoup(driver.page_source)

    return photo_lists