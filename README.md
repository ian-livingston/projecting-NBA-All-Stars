# Predicting Next Season's NBA All-Stars From Last Season's Data
## Project 3 (Metis bootcamp)

# Description
For my second solo data science project, I walked for the first of many times into the world of NBA analytics. My goal was to build a classification model with which to predict as closely as possible the 24 NBA All-Stars named in a season using any and all data available at the moment the previous season wraps. A few clarifications:
- As my data I used player-season combos ("Ray Allen, 2008") including and between 1999-00 and 2015-16 from two Kaggle datasets and supplementary web scraping.
- I decided to use the initial 24 All-Stars as my target positives. All-Stars named as injury replacements would not count as All-Stars, while initial selections, both starters and reserves, would count even if they ceded their spots due to injury, etc. I made this call to simplify my prediction efforts and to narrow in on the primary system(s) of All-Star selection and not the more subjective processes invoked when a replacement is need.
- Rookies, with no previous seasons of data, could not be counted. This is one of many compromises I hope to find a solve for at a later time.

# Features and Target Variables
My target variable was the label on an NBA player as an NBA All-Star next season (1) or not (0). My final model, an ensemble, was fit to be fed the following 11 features:

- "All-Star?" (previous year)
- "PTS/game"
- "AST/game"
- "Years from prime"
- "PER"
- "Trajectory" (homemade; explanation in notebook/.py file)
- "Adjusted TV market value * GS" (homemade; explanation in notebook/.py file)
- "TRB/game"
- "PTS+AST/game"
- "MP/game"
- "FT/game"

# Data Used
I used data scraped and sourced from:

- Kaggle
- stats.NBA.com
- basketball-reference.com
- sportbusinessdaily.com

# Tools Used
Additionally, I made use of the following Python modules in the course of this project:

- Scikit-learn
- Numpy
- Pandas
- BeautifulSoup
- Selenium
- Matplotlip
- Time
- Re
- Fuzzywuzzy
- imblearn
- XGBoost
- Tableau
- Flask

# Possible impacts
At the end of this time-warped project schedule my major takeaway is that the primary counting stats of PPG, APG and RBG are the drivers of performance I expected them to be. This is a little disappointing in that I wasn't able to feature engineer out of their gravity. Defense, meanwhile, was proven to be unwelcome or at least unnecessary at All-Star Weekend and meaning Rudy Gobert is doomed to be snubbed again and again.

Player age stood out early, in terms of intuition and in my simple plotting, as a central influence in the All-Star pipeline, but at this point I think it is more likely an indirect mesaurement of potential, trajectory, growth, or some other word like that. It remains difficult to project player development and peak, especially from afar. Data from college years and earlier might help uncover patterns in specific areas of the game, maybe even mapped to account for physical growth. It is additionally relevant in more advanced measures of player profile/star power. Data exists in the international game, at the college level, and (maybe?) at the high school/AAU level that could inform a better NBA model like mine.

Though I am unchanged in my belief that volume is a truer measure of All-Star potential than efficiency, I remain interested in deeper insights and take from my modeling efforts a few related thoughtlines to follow in the future. If a player gets minutes, and moreover spends much of them with the ball in his hands, he is on some track to see his All-Star chances rise. But to what extent do more minutes necessarily mean more production, and likewise to what extent does production earn a player more minutes? There's the Westbrook route to All-Star, but is there also a real Draymond route there on which a player can contributes in non-traditional, non-volume-heavy, non-iso-ball ways in the orbit of superstar power and winning? There's a lot I want to measure.

More areas of future focus:

- Rookies
- Team performance (maybe predicting in season)
- Coaching
- Player continuity on specific teams
- Changes in All-Star selcetion process (F/C -> F, new voting/coaches/media system)
- Changes in feature ranking over time (more important to shoot 3s now?)
- More exploration of profile, maybe using social media presence, shoe deals, sponsorships, national TV games, coverage by individual teams/NBA official/ESPN/etc.
- International fan voting, particularly for favorite or homegrown players
