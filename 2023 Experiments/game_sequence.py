import sportsdataverse
import numpy as np
from datetime import date, datetime, timedelta
from sportsipy.ncaab.teams import Teams
from pickle import dump
from tqdm import tqdm
from itertools import chain
import tensorflow as tf
from sys import argv

#print("getting data for season", argv[1])

CONFIG_YEAR = 2023#int(argv[1])
OUTPUT_FILE = "./games/game_{}.pkl"
SAVE_LOOKUP = False
LOAD_LOOKUP = False
RESTRICTED_KEYS = True
pg_keys = [
    'assists',#PG
    'blocks',#PG
    'defensive_rebounds',#PG
    'field_goal_attempts',#PG
    'field_goals',#PG
    'free_throw_attempts',#PG
    'free_throws',#PG
    'offensive_rebounds',#PG
    'opp_assists',#PG
    'opp_blocks',#PG
    'opp_defensive_rebounds',#PG
    'opp_field_goal_attempts',#PG
    'opp_field_goals',#PG
    'opp_free_throw_attempts',#PG
    'opp_free_throws',#PG
    'opp_offensive_rebounds',#PG
    'opp_personal_fouls',#PG
    'opp_points',#PG
    'opp_steals',#PG
    'opp_three_point_field_goal_attempts',#PG
    'opp_three_point_field_goals',#PG
    'opp_two_point_field_goal_attempts',#PG
    'opp_two_point_field_goals',#PG
    'opp_total_rebounds',#PG
    'opp_turnovers',#PG
    'personal_fouls',#PG
    'points',#PG
    'steals',#PG
    'three_point_field_goal_attempts',#PG
    'three_point_field_goals',#PG
    'two_point_field_goal_attempts',#PG
    'two_point_field_goals',#PG
    'total_rebounds',#PG
    'turnovers',#PG
    'win_percentage',#PG
]
valid_keys = [
    'assist_percentage',
    'block_percentage',
    'defensive_rebounds',
    'effective_field_goal_percentage',
    #'field_goal_attempts',#PG
    #'field_goal_percentage',
    #'field_goals',#PG
    'free_throw_attempt_rate',
    'free_throw_percentage',
    'free_throws_per_field_goal_attempt',##
    'offensive_rating',
    'offensive_rebound_percentage',
    'opp_assist_percentage',
    'opp_block_percentage',
    'opp_defensive_rebounds',#PG
    'opp_effective_field_goal_percentage',
    #'opp_field_goal_attempts',#PG
    #'opp_field_goal_percentage',
    #'opp_field_goals',#PG
    'opp_free_throw_attempt_rate',
    'opp_free_throw_percentage',
    'opp_free_throws_per_field_goal_attempt',##
    'opp_offensive_rebound_percentage',
    'opp_personal_fouls',#PG
    'opp_points',#PG
    'opp_steal_percentage',
    'opp_three_point_attempt_rate',
    'opp_three_point_field_goal_percentage',
    'opp_two_point_field_goal_attempts',#PG
    'opp_two_point_field_goal_percentage',
    'opp_two_point_field_goals',#PG
    'opp_total_rebound_percentage',
    'opp_true_shooting_percentage',
    'opp_turnover_percentage',
    'pace',
    'personal_fouls',#PG
    'points',#PG
    'simple_rating_system',
    'steal_percentage',
    'strength_of_schedule',
    'three_point_attempt_rate',
    'three_point_field_goal_percentage',
    'two_point_field_goal_attempts',#PG
    'two_point_field_goal_percentage',
    'two_point_field_goals',#PG
    'total_rebound_percentage',
    'true_shooting_percentage',
    'turnover_percentage',
] if RESTRICTED_KEYS else [
    'assist_percentage',
    'assists',#PG
    'block_percentage',
    'blocks',#PG
    'defensive_rebounds',
    'effective_field_goal_percentage',
    'field_goal_attempts',#PG
    'field_goal_percentage',
    'field_goals',#PG
    'free_throw_attempt_rate',
    'free_throw_attempts',#PG
    'free_throw_percentage',
    'free_throws',#PG
    'free_throws_per_field_goal_attempt',
    'offensive_rating',
    'offensive_rebound_percentage',
    'offensive_rebounds',#PG
    'opp_assist_percentage',
    'opp_assists',#PG
    'opp_block_percentage',
    'opp_blocks',#PG
    'opp_defensive_rebounds',#PG
    'opp_effective_field_goal_percentage',
    'opp_field_goal_attempts',#PG
    'opp_field_goal_percentage',
    'opp_field_goals',#PG
    'opp_free_throw_attempt_rate',
    'opp_free_throw_attempts',#PG
    'opp_free_throw_percentage',
    'opp_free_throws',#PG
    'opp_free_throws_per_field_goal_attempt',
    'opp_offensive_rebound_percentage',
    'opp_offensive_rebounds',#PG
    'opp_personal_fouls',
    'opp_points',
    'opp_steal_percentage',
    'opp_steals',#PG
    'opp_three_point_attempt_rate',
    'opp_three_point_field_goal_attempts',#PG
    'opp_three_point_field_goal_percentage',
    'opp_three_point_field_goals',#PG
    'opp_two_point_field_goal_attempts',#PG
    'opp_two_point_field_goal_percentage',
    'opp_two_point_field_goals',#PG
    'opp_total_rebound_percentage',
    'opp_total_rebounds',#PG
    'opp_true_shooting_percentage',
    'opp_turnover_percentage',
    'opp_turnovers',#PG
    'pace',
    'personal_fouls',
    'points',
    'simple_rating_system',
    'steal_percentage',
    'steals',#PG
    'strength_of_schedule',
    'three_point_attempt_rate',
    'three_point_field_goal_attempts',#PG
    'three_point_field_goal_percentage',
    'three_point_field_goals',#PG
    'two_point_field_goal_attempts',#PG
    'two_point_field_goal_percentage',
    'two_point_field_goals',#PG
    'total_rebound_percentage',
    'total_rebounds',#PG
    'true_shooting_percentage',
    'turnover_percentage',
    'turnovers',#PG
]
"""
teams = Teams(CONFIG_YEAR)#STANDARDIZE STATS TODO
invalid_teams = []
for team in list(teams):
    try:
        team.dataframe[pg_keys] = team.dataframe[pg_keys].values/team.games_played
    except:
        invalid_teams.append(team.name)
if len(invalid_teams) > 0:
    print("removing", *invalid_teams)
    teams._teams = list(filter(lambda e: e.name not in invalid_teams, teams._teams))
for column in valid_keys:
    try:
        mean = np.mean(teams.dataframes[column].values)
    except Exception as err:
        print(column)
        print(teams.dataframes[column].values)
        raise err
    std = np.std(teams.dataframes[column].values)
    for team in list(teams):
        team.dataframe[column] = (team.dataframe[column].values - mean)/std
sv = sportsdataverse.mbb.mbb_loaders.load_mbb_pbp([CONFIG_YEAR])
svstat = sportsdataverse.mbb.mbb_loaders.load_mbb_team_boxscore([CONFIG_YEAR])
"""
##SV add seconds since start
def base_seconds(period):
    period = int(period)
    if period == 1:
        return 20 * 60
    else:
        return 40 * 60 + (period - 2) * 5 * 60
"""
sv["seconds_since_start"] = np.vectorize(base_seconds)(sv.period.values) - 60 * sv.clock_minutes.values.astype("int32") - sv.clock_seconds.values.astype("int32")
"""
##SV TIME FIX
#for game_id in np.unique(sv.game_id.values):
#    sv[sv.game_id == game_id].values[0]["end_game_seconds_remaining"] = np.max(sv[sv.game_id == game_id].end_game_seconds_remaining.values)
#    sv[sv.game_id == game_id].values[0]["end_game_seconds_remaining"] = 0.0

#list(l)[0].schedule[0].dataframe.datetime.values[0]
def tdt(n):
    d = datetime.utcfromtimestamp(n.astype(datetime) * 1e-9)
    return date(d.year, d.month, d.day) + timedelta(1)

def match_team(sports_ref_team, svstat):
    #gs = [g.dataframe.date.values[0] for g in sports_ref_team.schedule]
    #idxs = [value.strftime("%a, %b %-d, %Y") in gs for value in svstat.game_date.values]
    #ids, count = np.unique(svstat[idxs].team_id.values, return_counts=True)
    dates = [tdt(game.dataframe.datetime.values[0]) for game in sports_ref_team.schedule]
    ids, count = np.unique(np.concatenate([svstat[svstat.game_date == d].team_id.values for d in dates], axis=0), return_counts=True)#predicated on the idea that no teams will share all their game dates
    #print(count[[svstat[svstat.team_id == idd].team_display_name.values[0] for idd in ids].index("Youngstown State Penguins")])
    #pds = [svstat[svstat.game_date == d] for d in dates]
    #pds = pd.concat([svs[svs.home_away == s.dataframe.location.values[0].upper()] for svs, s in zip(pds, sports_ref_team.schedule)])
    #ids, count = np.unique(pds.team_id.values, return_counts=True)#predicated on the idea that no teams will share all their game dates
    cidx = count >= np.max(count) - 5
    possible_ids = ids[cidx]
    possible_counts = count[cidx]
    if len(possible_ids) == 1:
        return possible_ids[0]
    #CHECKOUT PROVIDENCE, KENTUCKY, UMBC, NC STATE, USC, UC
    filtered_name = sports_ref_team.name.lower().replace("uab ", "birmingham ").replace("maryland-baltimore", "umbc").replace("ucf", "central florida").replace("southern california", "usc").replace("massachussetts", "mass").replace("mississippi", "miss").replace("north ", "").replace("northern ", "").replace("south ", "").replace("saint ", "").replace("university ", "").replace("st. ", "").replace("virginia commonwealth", "vcu").replace("connecticut", "uconn").replace("college ", "").replace(" of ", "").split("/")[0].split(" ")[0].split("-")[0].split("_")[0].split("'")[0].lower()
    possible_names = [(svstat[svstat.team_id == idd].team_slug.values[0] + svstat[svstat.team_id == idd].team_location.values[0] + svstat[svstat.team_id == idd].team_name.values[0] + svstat[svstat.team_id == idd].team_abbreviation.values[0] + svstat[svstat.team_id == idd].team_display_name.values[0] + svstat[svstat.team_id == idd].team_short_display_name.values[0]).lower() for idd in possible_ids]
    #possible_names = [[svstat[svstat.team_id == idd].team_display_name.values[0], svstat[svstat.team_id == idd].team_abbreviation.values[0], svstat[svstat.team_id == idd]..values[0]] for idd in possible_ids]
    #print(possible_names)
    try:
        name_filtered_idx = np.array([filtered_name in n.replace(" ", "").replace("'", "").lower() for n in possible_names]) == True
        s = np.sum(name_filtered_idx)
        if s == 1:
            return possible_ids[np.argmax(name_filtered_idx)]
        elif s == 0:
            assert False
        else:
            return possible_ids[name_filtered_idx][np.argmax(possible_counts[name_filtered_idx])]
    except:
        #print(possible_names)
        #print("could not find in possible names " + sports_ref_team.name)
        return None
"""
if LOAD_LOOKUP:
    with open("./name2id.txt", "r") as doc:
        name2id = eval(doc.read())
    team2id = {list(filter(lambda e: e.name == k, list(teams)))[0]: int(v) for k, v in name2id.items()}
else:
    invalid_teams = []
    for team in tqdm(list(teams), desc="matching"):
        try:
            team.schedule
        except:
            invalid_teams.append(team.name)
    if len(invalid_teams) > 0:
        print("removing", *invalid_teams)
        teams._teams = list(filter(lambda e: e.name not in invalid_teams, teams._teams))
    team2id = {team: match_team(team) for team in list(teams)}
    team2id = {k: int(v) for k, v in team2id.items() if v is not None}
    if SAVE_LOOKUP:
        with open("./name2id.txt", "w+") as doc:
            doc.write(repr({k.name: v for k, v in team2id.items()}))
id2team = {v: k for k, v in team2id.items()}
valid_ids = [v for _, v in team2id.items()]
#print(valid_ids)
"""
"""
invalid_keys = []
valid_keys = []
for c in list(list(teams)[0].dataframe.columns):
    try:
        int(list(teams)[0].dataframe[c].values.reshape(1).tolist()[0])
        if c not in invalid_keys:
            valid_keys.append(c)
    except:
        pass
"""

class SCSV(object):
    def __init__(self, year):
        self.year = year
        self.sv = sportsdataverse.mbb.mbb_loaders.load_mbb_pbp([year])
        self.modify_sv()
        self.svstat = sportsdataverse.mbb.mbb_loaders.load_mbb_team_boxscore([year])
        self.teams = self.build_teams_obj(self.year)
        self.match()
        self.keys = valid_keys
        self._games = []
        self._schedules = {}
        self.games()
        self._ids = np.unique([int(i.home_id) for i in self._games] + [int(i.away_id) for i in self._games])
        self.schedules()
        #self.make_index()
    
    def build_teams_obj(self, year):
        teams = Teams(year)#STANDARDIZE STATS TODO
        #teams._teams = [teams._teams[3], teams._teams[12]]
        invalid_teams = []
        for team in list(teams):
            try:
                team.dataframe[pg_keys] = team.dataframe[pg_keys].values/team.games_played
            except:
                invalid_teams.append(team.name)
        if len(invalid_teams) > 0:
            print("removing", *invalid_teams)
            teams._teams = list(filter(lambda e: e.name not in invalid_teams, teams._teams))
        for column in valid_keys:
            try:
                mean = np.mean(teams.dataframes[column].values)
            except Exception as err:
                print(column)
                print(teams.dataframes[column].values)
                raise err
            std = np.std(teams.dataframes[column].values)
            #for team in list(teams):
            #    team.dataframe[column] = (team.dataframe[column].values - mean)/std
        for team in tqdm(list(teams), desc="matching"):
            try:
                team.schedule
            except:
                invalid_teams.append(team.name)
        if len(invalid_teams) > 0:
            print("removing", *invalid_teams)
            teams._teams = list(filter(lambda e: e.name not in invalid_teams, teams._teams))
        return teams
    
    def match(self):
        team2id = {team: match_team(team, self.svstat) for team in list(self.teams)}
        self.team2id = {k: int(v) for k, v in team2id.items() if v is not None}
        self.id2team = {v: k for k, v in self.team2id.items()}
        self.valid_ids = [v for _, v in self.team2id.items()]
        #print(team2id, self.team2id, self.id2team, self.valid_ids)
    
    def modify_sv(self):
        self.sv["seconds_since_start"] = np.vectorize(base_seconds)(self.sv.period.values) - 60 * self.sv.clock_minutes.values.astype("int32") - self.sv.clock_seconds.values.astype("int32")
    
    def make_index(self):
        all_games = self.expanded()
        self.play_index = {play_type: n + 1 for n, play_type in enumerate(sorted(np.unique(np.concatenate([np.unique(i[0]) for i, _ in all_games]))))}
        self.score_index = {score_type: n + 1 for n, score_type in enumerate(sorted(np.unique(np.concatenate([np.unique(i[1]) for i, _ in all_games]))))}
    
    def __iter__(self):
        return self._games.__iter__()
    
    def index_types(self):
        return [play for play, _ in self.play_index.items()], [score for score, _ in self.score_index.items()]
    
    def __getitem__(self, i):
        #print(i, type(i))
        if type(i) in [slice]:
            if i.start is not None and i.stop is not None:
                key = lambda e: i.start <= e.game_date < i.stop
            elif i.start is not None:
                key = lambda e: i.start <= e.game_date
            elif i.stop is not None:
                key = lambda e: e.game_date < i.stop
        elif type(i) in [tuple]:
            int(i[0])
            #assert type(i[0]) in [int, float, str] and type(i[1]) in [slice]
            if i[1].start is not None and i[1].stop is not None:
                _key = lambda e: i[1].start <= e.game_date < i[1].stop
            elif i[1].start is not None:
                _key = lambda e: i[1].start <= e.game_date
            elif i[1].stop is not None:
                _key = lambda e: e.game_date < i[1].stop
            key = lambda e: int(i[0]) in [e.home_id, e.away_id] and _key(e)
        else:#if type(i) in [int, float, str]:
            key = lambda e: int(i) in [e.home_id, e.away_id]
        return list(filter(key, sorted(self.games(), key=lambda a: a.game_date)))
    
    def games(self, recalculate=False):
        if self._games == [] or recalculate:
            self._games = []
            for game_id in tqdm(np.unique(self.sv.game_id.values), desc="SV.games"):
                game = self.sv[self.sv.game_id == game_id].sort_values("seconds_since_start")
                if int(game.home_team_id.values[0]) in self.valid_ids and int(game.away_team_id.values[0]) in self.valid_ids:
                    try:
                        game_date = self.svstat[self.svstat.game_id == game_id].game_date.values[0]
                    except IndexError:
                        print("game_date for " + str(game_id) + " not found, trying teams search")
                        try:
                            game_date = list(filter(lambda e: e.location.lower() in ["home", "neutral"] and e.opponent_name == self.id2team[int(game.away_team_id.values[0])].name, list(self.id2team[int(game.home_team_id.values[0])].schedule)))[0].datetime
                            game_date = date(game_date.year, game_date.month, game_date.day)
                        except IndexError:
                            print("game_date for " + str(game_id) + " not found")
                    try:
                        self._games.append(Game(
                            int(game_id),
                            int(game.home_team_id.values[0]),
                            int(game.away_team_id.values[0]),
                            game.type_id.values.astype("int32"),
                            np.array([i if i is None else int(i) for i in game.team_id.values]),
                            game.score_value.values.astype("int32"),
                            game.seconds_since_start.values,
                            game_date,
                            self.id2team
                        ))
                    except Exception as err:
                        print(type(err), err, "prevented game", game_id, game_date, "from being added")
            self._games = list(sorted(self._games, key=lambda e: e.game_date))
        #print(len(self._games), self.valid_ids)
        return self._games
    
    def schedules(self, recalculate=False):
        if self._schedules == {} or recalculate:
            self._schedules = {}
            for _id in self._ids:
                self._schedules[_id] = self.get_team_dependent(_id)
        return self._schedules
    
    def get_team_dependent(self, team_id):
        return [[i.preset(team_id) for i in self[team_id, :g.game_date]] + [(g, g.preset(team_id))] for g in self[team_id]]
    
    def expanded(self):
        return [game.home_perspective() for game in self.games()] + [game.away_perspective() for game in self.games()]

    def windata(self):
        play_lookup = np.vectorize(lambda e: self.play_index[e])
        score_lookup = np.vectorize(lambda e: self.score_index[e])
        win_lookup = lambda e: [0, 1][e]
        expanded = self.expanded()
        #print(np.array(expanded[0][][0]).shape, np.array(expanded[0][1]).shape, np.array(expanded[0][2]).shape, np.array(expanded[0][3]).shape)
        data = zip(
            tf.keras.utils.pad_sequences([play_lookup(game[0]).tolist() for game, _ in expanded], padding="post"),
            tf.keras.utils.pad_sequences([score_lookup(game[1]).tolist() for game, _ in expanded], padding="post"),
            tf.keras.utils.pad_sequences([game[2] for game, _ in expanded], padding="post"),
            [game[3] for game, _ in expanded],
            [win_lookup(result) for _, result in expanded]
        )
        return [((p, s, t, st), w) for p, s, t, st, w in data]
    
    def asdata(self):
        """
        With Dependencies
        """
        all_games = chain(*[v for _, v in self._schedules.items()])
        return [[game.home_perspective() for game in game_deps] for game_deps in all_games] + [[game.away_perspective() for game in game_deps] for game_deps in all_games]

"""
class SV(object):
    def __init__(self):
        self.keys = valid_keys
        self._games = []
        self._schedules = {}
        self.games()
        self._ids = np.unique([int(i.home_id) for i in self._games] + [int(i.away_id) for i in self._games])
        self.schedules()
        self.make_index()
    
    def make_index(self):
        all_games = self.expanded()
        self.play_index = {play_type: n + 1 for n, play_type in enumerate(sorted(np.unique(np.concatenate([np.unique(i[0]) for i, _ in all_games]))))}
        self.score_index = {score_type: n + 1 for n, score_type in enumerate(sorted(np.unique(np.concatenate([np.unique(i[1]) for i, _ in all_games]))))}
    
    def __iter__(self):
        return self._games.__iter__()
    
    def index_types(self):
        return [play for play, _ in self.play_index.items()], [score for score, _ in self.score_index.items()]
    
    def __getitem__(self, i):
        #print(i, type(i))
        if type(i) in [slice]:
            if i.start is not None and i.stop is not None:
                key = lambda e: i.start <= e.game_date < i.stop
            elif i.start is not None:
                key = lambda e: i.start <= e.game_date
            elif i.stop is not None:
                key = lambda e: e.game_date < i.stop
        elif type(i) in [tuple]:
            int(i[0])
            #assert type(i[0]) in [int, float, str] and type(i[1]) in [slice]
            if i[1].start is not None and i[1].stop is not None:
                _key = lambda e: i[1].start <= e.game_date < i[1].stop
            elif i[1].start is not None:
                _key = lambda e: i[1].start <= e.game_date
            elif i[1].stop is not None:
                _key = lambda e: e.game_date < i[1].stop
            key = lambda e: int(i[0]) in [e.home_id, e.away_id] and _key(e)
        else:#if type(i) in [int, float, str]:
            key = lambda e: int(i) in [e.home_id, e.away_id]
        return list(filter(key, sorted(self.games(), key=lambda a: a.game_date)))
    
    def games(self):
        if self._games == []:
            for game_id in tqdm(np.unique(sv.game_id.values), desc="SV.games"):
                game = sv[sv.game_id == game_id].sort_values("seconds_since_start")
                if int(game.home_team_id.values[0]) in valid_ids and int(game.away_team_id.values[0]) in valid_ids:
                    try:
                        game_date = svstat[svstat.game_id == game_id].game_date.values[0]
                    except IndexError:
                        print("game_date for " + str(game_id) + " not found, trying teams search")
                        try:
                            game_date = list(filter(lambda e: e.location.lower() in ["home", "neutral"] and e.opponent_name == id2team[int(game.away_team_id.values[0])].name, list(id2team[int(game.home_team_id.values[0])].schedule)))[0].datetime
                            game_date = date(game_date.year, game_date.month, game_date.day)
                        except IndexError:
                            print("game_date for " + str(game_id) + " not found")
                    self._games.append(Game(
                        int(game_id),
                        int(game.home_team_id.values[0]),
                        int(game.away_team_id.values[0]),
                        game.type_id.values.astype("int32"),
                        np.array([i if i is None else int(i) for i in game.team_id.values]),
                        game.score_value.values.astype("int32"),
                        game.seconds_since_start.values,
                        game_date
                    ))
            self._games = list(sorted(self._games, key=lambda e: e.game_date))
        return self._games
    
    def schedules(self):
        if self._schedules == {}:
            for _id in self._ids:
                self._schedules[_id] = self.get_team_dependent(_id)
        return self._schedules
    
    def get_team_dependent(self, team_id):
        return [self[team_id, :g.game_date] + [g] for g in self[team_id]]
    
    def expanded(self):
        return [game.home_perspective() for game in self._games] + [game.away_perspective() for game in self._games]

    def windata(self):
        play_lookup = np.vectorize(lambda e: self.play_index[e])
        score_lookup = np.vectorize(lambda e: self.score_index[e])
        win_lookup = lambda e: [0, 1][e]
        expanded = self.expanded()
        #print(np.array(expanded[0][][0]).shape, np.array(expanded[0][1]).shape, np.array(expanded[0][2]).shape, np.array(expanded[0][3]).shape)
        data = zip(
            tf.keras.utils.pad_sequences([play_lookup(game[0]).tolist() for game, _ in expanded], padding="post"),
            tf.keras.utils.pad_sequences([score_lookup(game[1]).tolist() for game, _ in expanded], padding="post"),
            tf.keras.utils.pad_sequences([game[2] for game, _ in expanded], padding="post"),
            [game[3] for game, _ in expanded],
            [win_lookup(result) for _, result in expanded]
        )
        return [((p, s, t, st), w) for p, s, t, st, w in data]
    
    def asdata(self):
        all_games = chain(*[v for _, v in self._schedules.items()])
        return [[game.home_perspective() for game in game_deps] for game_deps in all_games] + [[game.away_perspective() for game in game_deps] for game_deps in all_games]
"""
class YSV(object):#GAMES DONT GET INDICIZED
    def __init__(self, years):
        self.years = years
        self.keys = valid_keys
        self.svs = {}
        for year in years:
            try:
                self.svs[year] = SCSV(year)
            except Exception as err:
                #raise err
                print(year, "failed")
        self.standardize_teams()
        self._games = []
        self.games(recalculate=True)
        self._schedules = {}
        self.schedules(recalculate=True)
        self.make_index()
    
    def standardize_teams(self):
        self.means = {}
        self.stds = {}
        for column in self.keys:
            _data = [[t.dataframe[column].values.tolist() for t in list(_sv.teams)] for _, _sv in self.svs.items()]
            data = []
            for i in _data:
                for g in i:
                    data += g
            mean = np.mean(data)
            self.means[column] = mean
            std = np.std(data)
            self.stds[column] = std
            for _, _sv in self.svs.items():
                for t in list(_sv.teams):
                    t.dataframe[column] = (t.dataframe[column].values - mean)/std

    def make_index(self):
        all_games = self.expanded()
        self.play_index = {play_type: n + 1 for n, play_type in enumerate(sorted(np.unique(np.concatenate([np.unique(i[0]) for i, _ in all_games]))))}
        self.score_index = {score_type: n + 1 for n, score_type in enumerate(sorted(np.unique(np.concatenate([np.unique(i[1]) for i, _ in all_games]))))}
    
    def __iter__(self):
        return self._games.__iter__()
    
    def games(self, recalculate=False):
        if self._games == [] or recalculate:
            self._games = []
            for _, i in self.svs.items():
                self._games += i.games(recalculate=recalculate)
        return self._games
    
    def index_types(self):
        return [play for play, _ in self.play_index.items()], [score for score, _ in self.score_index.items()]
    
    def __getitem__(self, i):
        #print(i, type(i))
        games = self.svs[i[0]].games
        i = i[1:]
        if len(i) == 1:
            i = i[0]
        if type(i) in [slice]:
            if i.start is not None and i.stop is not None:
                key = lambda e: i.start <= e.game_date < i.stop
            elif i.start is not None:
                key = lambda e: i.start <= e.game_date
            elif i.stop is not None:
                key = lambda e: e.game_date < i.stop
        elif type(i) in [tuple]:
            int(i[0])
            #assert type(i[0]) in [int, float, str] and type(i[1]) in [slice]
            if i[1].start is not None and i[1].stop is not None:
                _key = lambda e: i[1].start <= e.game_date < i[1].stop
            elif i[1].start is not None:
                _key = lambda e: i[1].start <= e.game_date
            elif i[1].stop is not None:
                _key = lambda e: e.game_date < i[1].stop
            key = lambda e: int(i[0]) in [e.home_id, e.away_id] and _key(e)
        else:#if type(i) in [int, float, str]:
            key = lambda e: int(i) in [e.home_id, e.away_id]
        return list(filter(key, sorted(games, key=lambda a: a.game_date)))
    
    def pad_games(self, max_len=440):
        gam = self.games()
        plays = tf.keras.utils.pad_sequences([g.home_masked_plays for g in gam], padding="post")
        plays = plays[:, :max_len]
        non_mas_plays = tf.keras.utils.pad_sequences([g.none_masked_plays for g in gam], padding="post")
        non_mas_plays = non_mas_plays[:, :max_len]
        times = tf.keras.utils.pad_sequences([g.times for g in gam], padding="post")
        times = times[:, :max_len]
        scores = tf.keras.utils.pad_sequences([g.home_masked_scores for g in gam], padding="post")
        scores = scores[:, :max_len]
        for g, p, n, t, s in zip(gam, plays, non_mas_plays, times, scores):
            g.home_masked_plays = p
            g.none_masked_plays = n
            g.times = t
            g.home_masked_scores = s
    
    def schedules(self, recalculate=False):
        if self._schedules == {} or recalculate:
            self._schedules = {}
            self._schedules = {year: self.svs[year].schedules(recalculate=recalculate) for year in self.svs}
        return self._schedules
    
    def get_team_dependent(self, team_id, year):
        return [self.svs[year][team_id, :g.game_date] + [g] for g in self.svs[year][team_id]]
    
    def expanded(self):
        return [game.home_perspective() for game in self.games()] + [game.away_perspective() for game in self.games()]

    def windata(self):
        play_lookup = np.vectorize(lambda e: self.play_index[e])
        score_lookup = np.vectorize(lambda e: self.score_index[e])
        win_lookup = lambda e: [0, 1][e]
        expanded = self.expanded()
        #print(np.array(expanded[0][][0]).shape, np.array(expanded[0][1]).shape, np.array(expanded[0][2]).shape, np.array(expanded[0][3]).shape)
        data = zip(
            tf.keras.utils.pad_sequences([play_lookup(game[0]).tolist() for game, _ in expanded], padding="post"),
            tf.keras.utils.pad_sequences([score_lookup(game[1]).tolist() for game, _ in expanded], padding="post"),
            tf.keras.utils.pad_sequences([game[2] for game, _ in expanded], padding="post"),
            [game[3] for game, _ in expanded],
            [win_lookup(result) for _, result in expanded]
        )
        return [((p, s, t, st), w) for p, s, t, st, w in data]
    
    def asdata(self):
        """
        With Dependencies
        """
        play_lookup = np.vectorize(lambda e: self.play_index[e])
        score_lookup = np.vectorize(lambda e: self.score_index[e])
        all_games = list(chain(*[vv for _, v in self.schedules().items() for _, vv in v.items()]))
        self.pad_games()
        games = self.games()
        games = [i.home_perspective for i in games] + [i.away_perspective for i in games]
        all_games = [n for n in all_games if len(n) > 1]
        labels = [i[-1] for i in all_games]
        all_games = [n[:-1] for n in all_games]
        _game_indices = [[games.index(i) for i in gams] for gams in all_games]#[[i.home_perspective()[0] for i in gams] for gams in all_games] + [[i.away_perspective()[0] for i in gams] for gams in all_games]
        _game_indices = tf.keras.utils.pad_sequences(_game_indices, value=-1, padding="post").tolist()
        #print(all_games[-1])
        #print(all_games[-1][0])
        shp = all_games[-1][0]()
        games += [lambda: ((np.zeros_like(shp[0][0]), np.zeros_like(shp[0][1]), np.zeros_like(shp[0][2]), (np.zeros_like(shp[0][3][0]), np.zeros_like(shp[0][3][1]))),)]
        games = [i()[0] for i in games]
        games = [(a.tolist(), b.tolist(), c.tolist(), (d.tolist(), e.tolist())) for (a, b, c, (d, e)) in games]
        game_indices = []
        for n, (label, sched) in enumerate(zip(labels, _game_indices)):
            idx = -1
            for n2, label2 in enumerate(labels):
                if label2[0] == label[0] and n2 != n:
                    idx = n2
                    break
            game_indices.append((sched, _game_indices[idx]))#np.expand_dims(_game_indices[idx], 0).tolist()))
        labels = [1 if label[1]()[-1] else 0 for label in labels]
        #max_len = np.max([len(i) for i in all_dat])
        #pad_dat = [[pad_val for _ in range(max_len)] for _ in all_dat]
        #for n, v in enumerate(all_dat):
        #    pad_dat[n][:len(v)] = v
        #pad_dat = [(np.concatenate([np.expand_dims(i[0], 0) for i in dat], axis=0), np.concatenate([np.expand_dims(i[1], 0) for i in dat], axis=0), np.concatenate([np.expand_dims(i[2], 0) for i in dat], axis=0), (np.concatenate([np.expand_dims(i[3][0], 0) for i in dat], axis=0), np.concatenate([np.expand_dims(i[3][1], 0) for i in dat], axis=0))) for dat in pad_dat]
        #need to group data tuples for batches
        #pad_dat = p_dat
        return game_indices, [(play_lookup(p).tolist(), score_lookup(s).tolist(), t, st) for p, s, t, st in games], labels
        #return [[game.home_perspective() for game in game_deps] for game_deps in all_games] + [[game.away_perspective() for game in game_deps] for game_deps in all_games]
"""
def _game_from_id(_sv, _svstat, game_id):
    game = _sv[_sv.game_id == game_id].sort_values("seconds_since_start")
    return Game(
        int(game_id),
        int(game.home_team_id.values[0]),
        int(game.away_team_id.values[0]),
        game.type_id.values.astype("int32"),
        game.team_id.values.astype("int32"),
        game.score_values,
        game.seconds_since_start.values,
        _svstat[_svstat.game_id == game_id].game_date.values[0]
    )
game_from_id = np.vectorize(partial(_game_from_id, sv, svstat))
"""

class Game(object):
    def __init__(self, game_id, home_id, away_id, play_types, committing_teams, scores, times, game_date, _id2team=None):
        self.id2team = _id2team #if _id2team is not None else id2team
        self.game_id = game_id
        self.game_date = game_date
        self.home_id = int(home_id)
        self.away_id = int(away_id)
        self.times = times
        #self.home_masks = committing_teams == self.home_id
        self.home_masks = np.vectorize(lambda e: {self.away_id: -1, self.home_id: 1, None: 0}[e])(committing_teams)
        self.unmasked_plays = np.array(play_types).astype("int32")
        self.home_masked_plays = self.unmasked_plays * self.home_masks
        self.none_masks = np.vectorize(lambda e: 1 if e == 0 else 0)(self.home_masks)
        self.none_masked_plays = self.unmasked_plays * self.none_masks
        #home_scores = np.squeeze(np.concatenate([np.array([0]), scores[:, 0]]))
        #home_scores = home_scores[1:] - home_scores[:-1]
        #away_scores = np.squeeze(np.concatenate([np.array([0]), scores[:, 1]]))
        #away_scores = away_scores[1:] - away_scores[:-1]
        self.home_masked_scores = self.home_masks * scores #home_scores - away_scores
        self.home_win = np.sum(self.home_masked_scores) > 0
        self._build_stats()
        #might need to get game result here
    
    def __str__(self):
        return " ".join([str(self.away_id), "at", str(self.home_id), "on", self.game_date.strftime("%a, %b %-d, %Y")])
    
    def _build_stats(self):
        self.home_stats = self.id2team[self.home_id].dataframe[valid_keys].values.flatten()#need to filter for dtype
        self.away_stats = self.id2team[self.away_id].dataframe[valid_keys].values.flatten()
    
    def home_perspective(self):
        #arrange stats, arrange scores, mask_play_types
        return (self.home_masked_plays + self.none_masked_plays, self.home_masked_scores, self.times, (self.home_stats, self.away_stats)), self.home_win
    
    def away_perspective(self):
        return (-1 * self.home_masked_plays + self.none_masked_plays, -1 * self.home_masked_scores, self.times, (self.away_stats, self.home_stats)), not self.home_win
    
    def preset(self, id):
        return self.home_perspective if self.home_id == id else self.away_perspective
#ADD MASKABLE WIN TERM IN DATASET
if __name__ == "__main__":
    s = YSV([2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010])
    game_indices, games, labels = s.asdata()
    #print(list(zip(*wd))[0])
    #wd = [((p.tolist(), s.tolist(), t.tolist(), (s1.tolist(), s2.tolist())), w) for ((p, s, t, (s1, s2)), w) in zip(*wd)]
    from json import dump
    with open("./scheduledata.json", "w+") as doc:
        dump({"indices": game_indices, "data": games, "labels": labels}, doc)
    with open("./scheduledata_stats.json", "w+") as doc:
        dump({"means": s.means, "stds": s.stds}, doc)
    with open("./scheduledata_lookups.json", "w+") as doc:
        dump({"plays": {int(k): int(v) for k, v in s.play_index.items()}, "scores": {int(k): int(v) for k, v in s.score_index.items()}}, doc)