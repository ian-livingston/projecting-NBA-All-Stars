from tqdm import tqdm
import numpy as np
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import pickle
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time, os

# For gathering All-Star fan voting data
# I eventually found and settled on using my Kaggle datasets and scrapped the scraping I did with this function
def get_all_star_votes(list_of_years):
    '''Takes a list of years as ints and returns data on NBA-Stars and All-Star voting in those years.
    Arguments: List
    Returns: Dictionary
    '''
    
    ua = UserAgent()
    user_agent = {'User-agent': ua.random}

    all_star_votes = {}
    for year in list_of_years:
        if year != 1999:
            year_all_stars_dict = {}
            url = "https://www.basketball-reference.com/allstar/NBA_{}_voting.html".format(year)
            response = requests.get(url, headers=user_agent)
            print("{} request status code: {}".format(str(year), str(response.status_code)))
            page = response.text
            soup = BeautifulSoup(page)
            whole_set = soup.find_all("div", class_="data_grid_group solo")

            # Getting starters
            starters = soup.find("div", class_="data_grid_group solo").find_all("div", id=re.compile("(all-star-starters).+"))
            starters_dict = {}
            for team in starters:
                conference = team.find("table").caption.text[:4]
                conference_dict = {}
                players = team.find("table").find_all("tr")
                for player in players:
                    player_dict = {}
                    player_details = player.find_all("td")
                    if len(player_details) == 3:
                        position = player_details[0].text
                        name = player_details[1].text
                        votes = int("".join(player_details[2].text.split(",")))
                        stats_page = "https://www.basketball-reference.com{}".format(player_details[1].a["href"])
                        conference_dict[name] = {"Position": position, "Votes": votes, "Player page": stats_page}
                year_all_stars_dict["{} starters".format(conference)] = conference_dict

            # Getting East votes
            east = whole_set[1]
            east_dict = {}
            for table in east.find_all("table"):
                position = table.caption.text
                players = table.find_all("tr")
                for player in players:
                    player_dict = {}
                    player_details = player.find_all("td")
                    if len(player_details) == 3:
                        name = player_details[1].text
                        votes = int("".join(player_details[2].text.split(",")))
                        stats_page = "https://www.basketball-reference.com{}".format(player_details[1].a["href"])
                        if year in [2013, 2014, 2015, 2016]:
                            if position == "Backcourt":
                                position = "BC"
                            else:
                                position = "FC"
                        east_dict[name] = {"Position": position, "Votes": votes, "Player page": stats_page}
            year_all_stars_dict["East full"] = east_dict

            # Getting West votes
            west = whole_set[2]
            west_dict = {}
            for table in east.find_all("table"):
                position = table.caption.text
                players = table.find_all("tr")
                for player in players:
                    player_dict = {}
                    player_details = player.find_all("td")
                    if len(player_details) == 3:
                        name = player_details[1].text
                        votes = int("".join(player_details[2].text.split(",")))
                        stats_page = "https://www.basketball-reference.com{}".format(player_details[1].a["href"])
                        if year in [2013, 2014, 2015, 2016]:
                            if position == "Backcourt":
                                position = "BC"
                            else:
                                position = "FC"
                        west_dict[name] = {"Position": position, "Votes": votes, "Player page": stats_page}
            year_all_stars_dict["West full"] = west_dict

            # Adding year to larger dict and pickling
            all_star_votes[year] = year_all_stars_dict
            all_star_votes_df = pd.DataFrame.from_dict(all_star_votes, orient="index")
            with open("all_star_votes.pickle", "wb") as to_write:
                pickle.dump(all_star_votes_df, to_write)
            print("Just pickled!")

    return all_star_votes_df


# For getting All-Star roster data
# Another function I scrapped but that can be used later
def get_all_star_rosters(list_of_years):
    '''Takes a list of years as ints and returns a df with All-Star fan voting numbers for all
    players receving votes in those years.

    Args: list (of years as ints)
    Returns: df
    '''

    all_star_rosters = {}
    ua = UserAgent()
    user_agent = {'User-agent': ua.random}

    for year in list_of_years:
        if year != 1999:
            url = "https://www.basketball-reference.com/allstar/NBA_{}.html".format(year)
            new_response = requests.get(url, headers=user_agent)
            print("Request status code: {}".format(str(new_response.status_code)))
            new_page = new_response.text
            soup = BeautifulSoup(new_page)

            player_list = []
            rosters = soup.find_all("div", class_="overthrow table_container")
            for table in rosters:
                for player in table.find_all("th", scope="row"):
                    player_name = player.find("a")
                    if player_name != None:
                        player_list.append(player_name.text)

            injury_notes = soup.find_all("ul", class_="page_index")

            # Getting injury notes
            for injury in injury_notes:
                player_name = soup.find("ul", class_="page_index").find("a").text.strip()
                note = injury.find("div").text.strip()
                player_list.append("{} ({})".format(player_name, note))

            all_star_rosters[year] = player_list
            all_star_rosters_df = pd.DataFrame.from_dict(all_star_rosters, orient="index")
            with open("all_star_rosters.pickle", "wb") as to_write:
                    pickle.dump(all_star_rosters_df, to_write)
            print("Just pickled!")

    return all_star_rosters_df


def get_teams_dict():
    '''Scrapes and parses two pages using requests and BeautifulSoup() and returns a dictionary 
    including all NBA teams active between 1999-2000 and present with abbreviations and TV market
    data from 2000-01 and 2001-02.

    Arguments: (none)
    Returns: dict
    '''
    
    # Getting TV market size in 2000-01 and 2001-02 (except in Canada)
    ua = UserAgent()
    user_agent = {'User-agent': ua.random}
    url = "https://www.sportsbusinessdaily.com/Daily/Issues/2001/08/14/Ratings-Research/NIELSEN-DMA-RANKINGS.aspx"
    response = requests.get(url, headers=user_agent)
    print("Request status code: {}".format(str(response.status_code)))
    page = response.text
    soup = BeautifulSoup(page)

    table = soup.find("article", class_="article normal row u-vr4").find_all("table")[1]
    markets = table.find_all("tr")[1:]
    market_dict = {}
    for market in markets:
        market_details = market.find_all("td")
        city_as_list = [word.strip() for word in market_details[2].text.split("      ")]
        if city_as_list.count("") > 0:
            city_as_list.remove("")
        name = " ".join(city_as_list)
        size01 = int("".join(market_details[3].text.split(',')).strip())
        size02 = int("".join(market_details[4].text.split(',')).strip())
        market_dict[name] = [size01, size02]

    # Matching teams to markets
    url = "https://www.sportsmediawatch.com/nba-market-size-nfl-mlb-nhl-nielsen-ratings/"
    response = requests.get(url, headers=user_agent)
    print("Request status code: {}".format(str(response.status_code)))
    page = response.text
    soup = BeautifulSoup(page)

    market_team_dict = {}
    markets_teams = soup.find("tbody").find_all("tr")
    for market_team in markets_teams[1:]:
        rows = market_team.find_all("td")
        teams = rows[3].text
        if teams == "Wolves":
            teams = "Timberwolves"
        if teams != "no team":
            market = rows[1].text
            if market == "New YorkCity":
                market = "New York"
            elif market == "Washington D.C.":
                market = "Washington, DC (Hagerstown)"
            elif market == "Cleveland-Akron":
                market = "Cleveland"
            elif market == "Portland":
                market = "Portland, OR"
            elif market == "Boston":
                market = "Boston (Manchester)"
            elif market == "Orlando-Daytona":
                market = "Orlando-Daytona Beach-Melborne"
            if "," in teams:
                team_list = teams.split(",")
                for team in team_list:
                    market_team_dict[team] = market
            else:
                market_team_dict[teams] = market

    # Getting team abbreviations (which are in main dataset) and adding those missing (see below)
    url = "https://en.wikipedia.org/wiki/Wikipedia:WikiProject_National_Basketball_Association/National_Basketball_Association_team_abbreviations"
    response = requests.get(url, headers=user_agent)
    print("Request status code: {}".format(str(response.status_code)))
    page = response.text
    soup = BeautifulSoup(page)

    team_abbrev_dict = {}
    teams_abbrevs = soup.find("table").find_all("tr")
    for team in teams_abbrevs[1:]:
        abbrev = team.find_all("td")[0].text.strip()
        team_name = team.find_all("td")[1].text.strip()
        team_abbrev_dict[abbrev] = team_name

    # Adding: TOR, CHH, CHO, VAN, NOK, NJN, SEA, NOH
    team_abbrev_dict["CHH"] = "Charlotte Hornets"
    team_abbrev_dict["CHO"] = "Charlotte Hornets"
    team_abbrev_dict["NOH"] = "New Orleans Hornets"
    team_abbrev_dict["NOK"] = "New Orleans/Oklahoma City Hornets"
    team_abbrev_dict["NJN"] = "New Jersey Nets"
    team_abbrev_dict["SEA"] = "Seattle Supersonics"
    team_abbrev_dict["TOR"] = "Toronto Raptors"
    team_abbrev_dict["VAN"] = "Vancouver Grizzlies"

    # Compiling this all so far in one dict with abbreviations as keys
    new_dict = {}
    for key, value in team_abbrev_dict.items():
        if key == 'NOH':
            new_dict["NOH"] = {"Full name": "New Orleans Hornets", "Market": "New Orleans"}
        elif key == "NOK":
            new_dict["NOK"] = {"Full name": "New Orleans/Oklahoma City Hornets", "Market": "Oklahoma City"}
        elif key == "TOR":
            new_dict["TOR"] = {"Full name": "Toronto Raptors", "Market": "Toronto"}
        elif key == "VAN":
            new_dict["VAN"] = {"Full name": "Vancouver Grizzlies", "Market": "Vancouver"}
        elif key == "SEA":
            new_dict["SEA"] = {"Full name": "Seattle Supersonics", "Market": "Seattle-Tacoma"}
        else:
            for key2, value2 in market_team_dict.items():
                if key2 in value:
                    new_dict[key] = {"Full name": value, "Market": value2}

    # Adding market size to create full dict
    full_team_dict = {}
    for key, value in tqdm(new_dict.items()):
        if value["Market"] in market_dict:
            value_dict = value
            value_dict["2000-01 TV market size"] = market_dict[value["Market"]][0]
            value_dict["2001-02 TV market size"] = market_dict[value["Market"]][1]
            full_team_dict[key] = value_dict
        # This is essentially Canada, as I couldn't find TV market data to fit Nielsen's
        else:
            value_dict = value
            value_dict["2000-01 TV market size"] = np.NaN
            value_dict["2001-02 TV market size"] = np.NaN
            full_team_dict[key] = value_dict
    
    return full_team_dict

# For ensuring names in different datasets match
def name_cleaner(name):
    '''Takes a player name and "cleans" it to match a style determined appropriate by me
    as an NBA fan and over the course of this project and mostly to ensure the name matches
    the NBA's recorded version of that player's name.

    Arguments: string
    Returns: string
    '''

    name = "".join(name.split("."))

    if name == "Nene Hilario":
        name = "Nene"
    elif name == "Luc Mbah":
        name = "Luc Mbah a Moute"
    elif name == "Roger Mason":
        name = "Roger Mason Jr."
    elif name == "Keith Van":
        name = "Keith Van Horn"
    elif name == "Ronald Murray":
        name = "Flip Murray"
    elif name == "Nick Van":
        name = "Nick Van Exel"
    elif name == "John Lucas":
        name = "John Lucas III"
    elif name == "Marcus Morris":
        name = "Marcus Morris Sr."
    elif name == "Wang Zhizhi":
        name = "Wang Zhi-zhi"

    return name


# For getting main df
def get_main_data(master_df, all_stars2000_2016, full_team_dict):
    '''Returns a df with starting data for NBA players from 1999-2000 season through 2015-16 season. 
    In the process unpickles a dict of city data, a dict of NBA player stats, and a df of All-Star data.
    
    Arguments: df, df, dict
    Returns: df
    '''

    # Pulling out a subset of full dataset to match the seasons covered in the other
    master_df = master_df[(master_df["Year"] >= 2000.0) & (master_df["Year"] <= 2016.00)].drop(columns="Unnamed: 0")
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

    # Dropping 2015-16 season (to be used later as a potential extra holdout/test season 1)
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
# This version does not account for guys who are not rookies in 2000 (first year of data)
def tv_market_cumulative(ordered_df):
    '''Takes an ordered df and adds an "Adjusted TV market value" column calculated from existing columns.

    Arguments: df
    Returns: df (the same one with additional columns)
    '''

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
                    same_guy = ordered_df.iloc[x]["Player"] == ordered_df.iloc[x-1]["Player"]
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
    '''Takes an ordered df and adds a "Trajectory" column calculated from existing columns.

    Arguments: df
    Returns: df (the same one with additional columns)
    '''

    projected_changes = []
    for i in tqdm(range(1, ordered_df.shape[0])):
        x = i
        current_guy = ordered_df.iloc[x]["Player"]
        previous_guy = ordered_df.iloc[x-1]["Player"]
        if current_guy == previous_guy:
            same_guy = True
            change_vector = []
            while same_guy == True:
                this_year_stat = ordered_df.iloc[x]["PTS/game"]
                last_year_stat = ordered_df.iloc[x-1]["PTS/game"]
                stat_change = this_year_stat - last_year_stat
                change_vector.append(stat_change)
                x -= 1
                same_guy = ordered_df.iloc[x]["Player"] == ordered_df.iloc[x-1]["Player"]
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
    '''Takes a list of seasons identified by the year of the respective NBA Finals, scrapes
    clutch stats for each player and returns them in a df.

    Arguments: list of years as ints
    Returns: df
    '''

    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    
    chromedriver = "/Applications/chromedriver"
    os.environ["webdriver.chrome.driver"] = chromedriver
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

    return clutch_stats_df


# For making a column with values reflecting whether a guy has previously made 0, 1-3, 4-7, or 8+ All-Star Games
# This currently cannot account for players who made All-Star Games before 2000-01 season
# First function is for binning returned years
def all_star_history_bins(single_year_value):
    '''Takes an int representing the number of previous All-Star Games a player has made and converts it to a string
    representing one of four allocated bins. For use in a lambda function.

    Arguments: int
    Returns: string
    '''

    if single_year_value >= 8:
        all_star_bin = "8+"
    elif single_year_value >= 4:
        all_star_bin = "4-7"
    elif single_year_value >= 1:
        all_star_bin = "1-3"
    else:
        all_star_bin = "0"
    
    return all_star_bin


# For creating a column of binned All-Star history, to later be dummied
# For use with the above function
def get_past_all_star_games(ordered_df):
    '''Takes an ordered df and creates a new column binning the number of All-Star Games a player 
    has played in. 

    Args: df
    Returns: df
    '''

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


# For getting links to all NBA player photos, which I didn't end up using
def get_photo_links(list_of_seasons):
    '''Takes a list of ints (years) and scrapes via Selenium the link to each player profile pic URL,
    to be used (or not) to download thousands of player photos in some later, streamlined process I 
    haven't yet determined.

    Arguments: list (of ints representing years)
    Returns: list of strings (links)
    '''

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


# For scraping clutch minutes, points, assists, and fantasy points (NBA's scoring system) from stats.NBA.com
def get_clutch_stats(list_of_seasons):
    '''Takes a list of years as ints and scrapes using Selenium clutch stats in those seasons, returning the same df
    with clutch stats added in new columns.

    Arguments: list (of years as ints)
    Returns: df
    '''

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

    return clutch_stats_df


# For adding clutch stat columns using fuzzywuzzy
def get_clutch(ordered_df):
    '''Takes a df and adds columns full of clutch stats.

    Arguments: df
    Returns:
    '''

    from fuzzywuzzy import process

    clutch_guys = []
    clutch_minutes = []
    clutch_points = []
    clutch_assists = []
    clutch_fantasy_points = []
    for year in ordered_df["Year"].unique():
        year_df = ordered_df[ordered_df["Year"] == year]
        year_players = list(ordered_df["Player"])
        year_clutch_df = clutch_df[clutch_df["Year"] == year]
        year_clutch_players = list(clutch_df[clutch_df["Year"] == year]["Player"])
        print(year, year_clutch_players)
        for guy in year_players:
            try:
                processed_guy = process.extractOne(guy, year_clutch_players)[0]
            except TypeError:
                clutch_guys.append(guy)
                clutch_minutes.append(0)
                clutch_points.append(0)
                clutch_assists.append(0)
                clutch_fantasy_points.append(0)
            if process.extractOne(guy, year_clutch_players)[1] < 50:
                print(guy, processed_guy)
            clutch_guys.append(guy)
            minutes = year_clutch_df[year_clutch_df["Player"] == processed_guy]["Clutch minutes"]
            clutch_minutes.append(minutes)
            points = year_clutch_df[year_clutch_df["Player"] == processed_guy]["Clutch points"]
            clutch_points.append(points)
            assists = year_clutch_df[year_clutch_df["Player"] == processed_guy]["Clutch assists"]
            clutch_assists.append(assists)
            fantasy_points = year_clutch_df[year_clutch_df["Player"] == processed_guy]["Clutch fantasy points"]
            clutch_fantasy_points.append(fantasy_points)
    
    ordered_df["Clutch name", "Clutch minutes", "Clutch points", "Clutch assists", "Clutch fantasy points"] = clutch_guys, clutch_minutes, clutch_points, clutch_assists, clutch_fantasy_points

    return ordered_df


# Adding column "Adjusted All-Star?"
def get_adjusted_all_star_games(ordered_df):
    '''Takes a df and creates a new column and target variable reflecting whether a player was an All-Star
    by way of the core selection process and not as an injury replacement.

    Args: df
    Returns: df
    '''

    selection_list = []
    for value in tqdm(ordered_df["Selection process"]):
        selection_list.append(str(value).split('\n')[0])

    non_replacement_all_stars = []
    for i in tqdm(range(ordered_df.shape[0])):
        player = ordered_df.iloc[i]["Player"]
        all_star = ordered_df.iloc[i]["All-Star?"]
        if (all_star == 1) and "Replacement" not in selection_list[i]:
            non_replacement_all_stars.append(1)
            print(player)
        else:
            non_replacement_all_stars.append(0)

    ordered_df["Adjusted All-Star?"] = non_replacement_all_stars
    
    return ordered_df