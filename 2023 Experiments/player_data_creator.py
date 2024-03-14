from sportsipy.ncaab.teams import Teams
from itertools import chain
from json import dump
import numpy as np
from tqdm import tqdm

keys = ['class', 'height', 'weight'] + ['games_started', 'mp_per_g', 'fg_per_g', 'fga_per_g', 'fg_pct', 'fg2_per_g', 'fg2a_per_g', 'fg2_pct', 'fg3_per_g', 'fg3a_per_g', 'fg3_pct', 'ft_per_g', 'fta_per_g', 'ft_pct', 'orb_per_g', 'drb_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g']

def team_from_name(teams, name):
    try:
        return list(filter(lambda e: e.name == name, list(teams)))[0]
    except:
        return None#print(name, name in [i.name for i in list(teams)])

#def opponent_from_game(teams, game):
#    return list(filter(lambda e: any([for g in list(e.schedule)]), list(teams)))[0]

def calc_player_stats(df, name):
    return [df[name][key] for key in keys]

class PlayerDataHandler(object):
    def __init__(self, years):
        self.teams = [Teams(i) for i in tqdm(years, desc="years")]
        self.games = list(chain(*[list(chain(*[[(g, team, t) for g in list(team.schedule)] for team in tqdm(list(t), desc="t")])) for t in self.teams]))#FILTER THIS FOR TYPE NCAA FOR ONLY TOURNEY DATA
        self.labels = [{"win": 1, "loss": 0}[i[0].result.lower()] for i in tqdm(self.games, desc="self.games")]#[1 if game[0].result.lower == "win" else 0 for game in tqdm(self.games, desc="self.games")]
        print(self.labels[:10])
        self.op_stats = []
        for team in tqdm([team_from_name(game[2], game[0].opponent_name) for game in self.games], desc="op_stats"):
            r = []
            if team is not None:
                for player in list(team.roster.all_players.keys()):
                    try:
                        r.append(calc_player_stats(team.roster.all_players, player))
                    except Exception as err:
                        print(type(err), err, player, team, "failed")
            self.op_stats.append(r)
        self.me_stats = []
        for team in tqdm([game[1] for game in self.games], desc="me_stats"):
            r = []
            for player in list(team.roster.all_players.keys()):
                try:
                    r.append(calc_player_stats(team.roster.all_players, player))
                except Exception as err:
                    print(type(err), err, player, team, "failed")
            self.me_stats.append(r)
        #self.me_stats = [[calc_player_stats(player.dataframe, team) for player in team.roster.players] for team in tqdm([game[1] for game in self.games], desc="me_stats")]
        #self.op_stats = [[calc_player_stats(player.dataframe, team) for player in team.roster.players] for team in tqdm([team_from_name(game[1], game[0].opponent_name) for game in self.games], desc="op_stats")]
        self.together = zip(self.me_stats, self.op_stats)
        self.me_stats = []
        self.op_stats = []
        for t in self.together:
            t = (list(filter(lambda e: sum(e) > 0, t[0])), list(filter(lambda e: sum(e) > 0, t[1])))
            if len(t[0]) >= 5 and len(t[1]) >= 5:
                self.me_stats.append(t[0])
                self.op_stats.append(t[1])
        self.players = list(chain(*(self.me_stats + self.op_stats)))#list(chain(*list(chain(*(self.me_stats + self.op_stats)))))
        self.means = np.mean(self.players, axis=0)
        self.stds = np.std(self.players, axis=0)
        self.me_stats = [((np.array(ar) - self.means)/self.stds).tolist() for ar in self.me_stats]
        self.op_stats = [((np.array(ar) - self.means)/self.stds).tolist() for ar in self.op_stats]
        self.max_len = np.max([len(i) for i in self.me_stats + self.op_stats])
        self.pad_val = [0 for _ in range(len(keys))]
        self.me_stats = [i + [self.pad_val for _ in range(self.max_len - len(i))] for i in self.me_stats]
        self.op_stats = [i + [self.pad_val for _ in range(self.max_len - len(i))] for i in self.op_stats]
        self.means = self.means.tolist()
        self.stds = self.stds.tolist()
        self.examples = list(zip(list(zip(self.me_stats, self.op_stats)), self.labels))

if __name__ == "__main__":
    #d = PlayerDataHandler([2023])
    d = PlayerDataHandler([2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010])
    #print(d.examples[:10])
    with open("./playerdata.json", "w+") as doc:
        dump(d.examples, doc)
    with open("./playerdata_stats.json", "w+") as doc:
        dump({"means": d.means, "stds": d.stds}, doc)

"""PLAYER BASED
self.teams = [Teams(i) for i in tqdm(years, desc="years")]
        self.games = list(chain(*[list(chain(*[[(g, team, t) for g in list(team.schedule)] for team in tqdm(list(t), desc="t")])) for t in self.teams]))
        self.labels = [1 if game[0].result.lower == "win" else 0 for game in tqdm(self.games, desc="self.games")]
        self.me_stats = []
        for team in tqdm([game[1] for game in self.games], desc="me_stats"):
            r = []
            for player in team.roster.players:
                try:
                    r.append(calc_player_stats(player.dataframe, team))
                except Exception as err:
                    print(type(err), err, player, team, "failed")
            self.me_stats.append(r)
        #self.me_stats = [[calc_player_stats(player.dataframe, team) for player in team.roster.players] for team in tqdm([game[1] for game in self.games], desc="me_stats")]
        self.op_stats = []
        for team in tqdm([team_from_name(game[1], game[0].opponent_name) for game in self.games], desc="op_stats"):
            r = []
            for player in team.roster.players:
                try:
                    r.append(calc_player_stats(player.dataframe, team))
                except Exception as err:
                    print(type(err), err, player, team, "failed")
            self.op_stats.append(r)
        #self.op_stats = [[calc_player_stats(player.dataframe, team) for player in team.roster.players] for team in tqdm([team_from_name(game[1], game[0].opponent_name) for game in self.games], desc="op_stats")]
        self.together = zip(self.me_stats, self.op_stats)
        self.me_stats = []
        self.op_stats = []
        for t in self.together:
            t = (list(filter(lambda e: sum(e) > 0, t[0])), list(filter(lambda e: sum(e) > 0, t[1])))
            if len(t[0]) >= 5 and len(t[1]) >= 5:
                self.me_stats.append(t[0])
                self.op_stats.append(t[1])
        self.players = list(chain(*list(chain(*(self.me_stats + self.op_stats)))))
        self.means = np.mean(self.players, axis=0)
        self.stds = np.std(self.players, axis=0)
        self.examples = list(zip(list(zip(((np.array(self.me_stats) - self.means)/self.stds).tolist(), ((np.array(self.op_stats) - self.means)/self.stds).tolist())), self.labels))
        ###PAD EXAMPLES
        self.max_len = np.max([len(i) for i in self.me_stats + self.op_stats])
        self.pad_val = [0 for _ in range(len(keys))]
        self.me_stats = [i + [self.pad_val for _ in range(self.max_len - len(i))] for i in self.me_stats]
        self.op_stats = [i + [self.pad_val for _ in range(self.max_len - len(i))] for i in self.op_stats]
        self.means = self.means.tolist()
        self.stds = self.stds.tolist()
"""