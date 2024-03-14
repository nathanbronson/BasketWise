from copy import copy
from datetime import datetime, timedelta
from tqdm import tqdm
from math import log10, ceil, floor
import numpy as np
from pickle import dump, load
import os
from glob import glob
import sportsdataverse
import pandas as pd

GRADIENT_SWITCH_THRESH = .9
PROP = False
NOW = datetime.now()
LOAD_TEAMS = False
RESTORE = False
TOP = None#360
NORMALIZE_DIFS = False

method = ["tournament", "matchups", "save"][2]
plot = False #@param {type:"boolean"}
equal01 = False #@param {type:"boolean"}
addpg = False #@param {type:"boolean"}
replicate = False #@param {type:"boolean"}
secondrd = False #@param {type:"boolean"}
configyear = 2023 #@param {type:"integer"}
samplesize = 900 #@param {type:"integer"}
use_bank = False #@param {type:"boolean"}
team_bank = [
  "Gonzaga",
  "Georgia State",
  "Boise State",
  "Memphis"
]

def construct_difs(prop=PROP, top=TOP, normalize=NORMALIZE_DIFS):
    difs = []
    lens = []
    games = []
    game2teams2int = {}
    sv = sportsdataverse.mbb.mbb_loaders.load_mbb_pbp([configyear])
    #sv_events = [sv[sv.game_id == gid] for gid in np.unique(sv.game_id.values)]
    #sv_games = [sportsdataverse.mbb.mbb_pbp.espn_mbb_pbp(gid) for gid in np.unique(sv.game_id.values)]
    teams = np.unique(sv.home_team_name.values.tolist() + sv.away_team_name.values.tolist()).tolist()
    team_counts = [np.sum(sv.home_team_name.values == team) + np.sum(sv.away_team_name.values == team) for team in teams]
    teams = np.array(sorted(teams, key=lambda e: team_counts[teams.index(e)], reverse=True)[:top])
    start_time = np.max(sv.start_game_seconds_remaining)
    for team in tqdm(teams.tolist()):
        difs.append([])
        games.append([])
        hg = np.unique(sv[sv.home_team_name == team].game_id.values)
        ag = np.unique(sv[sv.away_team_name == team].game_id.values)#list(filter(lambda e: e.opponent_name in names, filter(lambda c: c.points_for is not None, filter(lambda e: e.datetime < NOW, team.schedule))))#g = list(filter(lambda e: e.opponent_name in [f.name for f in list(TEAMS)], filter(lambda c: c.points_for is not None, filter(lambda e: e.datetime < NOW, team.schedule))))
        for id in hg:
            if id in game2teams2int:
                difs[-1].append(game2teams2int[id][team] - game2teams2int[id][list(filter(lambda e: e != team, game2teams2int[id].keys()))[0]])
            else:
                game = sv[sv.game_id == id]
                if game.away_team_name.values[0] not in teams.tolist():
                    continue
                #s = np.concatenate([np.array([start_time]), game.start_game_seconds_remaining.values])
                #delta_t = s[:-1] - s[1:]
                #delta_t = delta_t/np.sum(delta_t)
                hts = np.max(game.home_score.values).astype("float32")#np.sum(game.home_score.values * delta_t) * 2
                ats = np.max(game.away_score.values).astype("float32")#np.sum(game.away_score.values * delta_t) * 2
                game2teams2int[id] = {team: hts, game.away_team_name.values[0]: ats}
                difs[-1].append((hts - ats)/(1 if not prop else hts + ats))
            games[-1].append([team, list(filter(lambda e: e != team, game2teams2int[id].keys()))[0]])
        for id in ag:
            if id in game2teams2int:
                difs[-1].append(game2teams2int[id][team] - game2teams2int[id][list(filter(lambda e: e != team, game2teams2int[id].keys()))[0]])
            else:
                game = sv[sv.game_id == id]
                if game.home_team_name.values[0] not in teams.tolist():
                    continue
                #s = np.concatenate([np.array([start_time]), game.start_game_seconds_remaining.values])
                #delta_t = s[:-1] - s[1:]
                #delta_t = delta_t/np.sum(delta_t)
                hts = np.max(game.home_score.values).astype("float32")#np.sum(game.home_score.values * delta_t) * 2
                ats = np.max(game.away_score.values).astype("float32")#np.sum(game.away_score.values * delta_t) * 2
                game2teams2int[id] = {team: ats, game.home_team_name.values[0]: hts}
                difs[-1].append((ats - hts)/(1 if not prop else hts + ats))
            games[-1].append([team, list(filter(lambda e: e != team, game2teams2int[id].keys()))[0]])
        lens.append(len(difs[-1]))
        games[-1] = np.array(games[-1])
    l = max(lens)
    _difs = []
    for a in difs:
        _difs.append(np.pad(a, (0, l - len(a) + 1), constant_values=np.nan))
    difs = np.array(_difs)
    _games = []
    for a in games:
        _games.append(np.pad(a, ((0, l - len(a) + 1), (0, 0)), constant_values=np.nan))
    games = np.array(_games)
    #print(games.shape, difs.shape, teams.shape)
    ndifs = np.nan_to_num(difs)
    if normalize:
        return (difs - np.mean(ndifs))/np.std(ndifs), games, teams
    else:
        return difs, games, teams

def points_SGD(descent_thresh=1e-12, lr0=1000, gradient_switch_thresh=GRADIENT_SWITCH_THRESH, prop=PROP):
    difs, games, teams = construct_difs(prop=prop)
    weights = np.random.rand(len(teams))
    l = weights.shape[0]
    t = teams.tolist()
    points_indices_0 = []
    points_indices_1 = []
    for team in games:
        #print(team)
        points_indices_0.append([])
        points_indices_1.append([])
        for game in team:
            #print(game)
            try:
                points_indices_0[-1].append(t.index(game[0]))
            except:
                points_indices_0[-1].append(-1)
            try:
                points_indices_1[-1].append(t.index(game[1]))
            except:
                points_indices_1[-1].append(-1)
            #print(points_indices_0[-1][-1], points_indices_1[-1][-1])
    points_indices_0 = np.tile(np.array(points_indices_0, dtype=np.int_), (l, 1, 1))
    points_indices_1 = np.tile(np.array(points_indices_1, dtype=np.int_), (l, 1, 1))
    #print(points_indices_0, points_indices_1, points_indices_0.shape, points_indices_1.shape)
    #points_indices = np.array([points_indices_0, points_indices_1])
    mask_indices = np.repeat(np.expand_dims(np.repeat(np.arange(l).reshape((l, 1)), points_indices_0.shape[-1], axis=1), 1), l, axis=1)#np.repeat(np.arange(l).reshape((l, 1)), (points_indices_0.shape[0] * points_indices_0.shape[1] * points_indices_0.shape[2]), axis=1).reshape(points_indices_0.shape)#np.repeat(np.expand_dims(np.repeat(np.arange(l).reshape((l, 1)), points_indices_0.shape[-1], axis=1), axis=0), l, axis=0)#mask_indices = np.repeat(np.arange(l).reshape((l, 1)), len(points_indices_0[0]), axis=1)
    #print(mask_indices, mask_indices.shape)
    #return
    mask = np.zeros((l, l + 1), dtype=np.float32)
    for i in range(l):
        mask[i, i] = .5
        mask[i, -1] = np.nan
    #np.concatenate((mask, np.expand_dims(np.array([np.nan for _ in range(l)]), axis=0)), axis=0)
    weights_record = [weights]
    ite = 0
    lr = lr0
    vals = np.array([])
    gradients = []
    tdif = np.tile(difs, (l, 1, 1))
    pi0 = points_indices_0[0]
    pi1 = points_indices_1[0]
    while True:
        try:
            tw = np.concatenate([weights_record[-1], np.zeros((1))])
            vals = np.concatenate([vals, [np.sum(np.abs(np.nan_to_num(difs - (tw[pi0] - tw[pi1]))))]])
            #weights_diff = weights[points_indices_0] - weights[points_indices_1]
            w = weights_record[np.argmin(vals)]
            tw = np.tile(np.concatenate([w, np.zeros((1))]), (l, 1))
            _g = (tw + mask)
            _l = (tw - mask)
            #weight_diffs_g = _g[mask_indices, points_indices_0] - _g[mask_indices, points_indices_1] - tdif
            #weight_diffs_l = _l[mask_indices, points_indices_0] - _l[mask_indices, points_indices_1] - tdif
            #gradient_breakdown = np.nan_to_num(np.abs(_g[mask_indices, points_indices_0] - _g[mask_indices, points_indices_1] - tdif) - np.abs(_l[mask_indices, points_indices_0] - _l[mask_indices, points_indices_1] - tdif))
            gradient = np.sum(np.sum(np.nan_to_num(np.abs(_g[mask_indices, points_indices_0] - _g[mask_indices, points_indices_1] - tdif) - np.abs(_l[mask_indices, points_indices_0] - _l[mask_indices, points_indices_1] - tdif)), axis=1), axis=1)
            gradients.append(gradient)
            #adjustments = -lr * np.sum(np.sum(np.nan_to_num(np.abs(_g[mask_indices, points_indices_0] - _g[mask_indices, points_indices_1] - tdif) - np.abs(_l[mask_indices, points_indices_0] - _l[mask_indices, points_indices_1] - tdif)), axis=1), axis=1)
            weights_record.append(w + (-lr * gradient))
            #print(gradient_breakdown, gradient_breakdown.shape)
            #return# - (tw - mask)[np.arange(l), ]
            #weights_diff = (tw + mask)[points_indices_0] - (tw - mask)[points_indices_1]
            ite += 1
            print(ite, vals[-1])
            if vals.shape[0] >= 2:
                if vals[-1] > np.min(vals) or vals[-1] == vals[-2]:
                    #weights = weights_record[np.argmin(vals)]
                    if lr >= descent_thresh:
                        loglr = log10(lr)
                        lr -= 10 ** (ceil(loglr) - 1)
                        lr = float(format(lr, ".{}f".format(str(2 + abs(floor(loglr))))))
                        print("LR", lr)
                    else:
                        return weights_record[np.argmin(vals)].tolist(), teams
                    continue
                """
                u = np.unique(-1 * np.sign(gradients[-1]) == np.sign(gradients[-2]), return_counts=True)
                if True in u[0] and {i : n for i, n in zip(*u)}[True]/sum(u[1]) >= gradient_switch_thresh:#if vals[-2] - vals[-1] <= descent_thresh * lr ** .5:#5 ** log10(lr) INSTEAD DETECT OVERSHOOT WHERE GRADIENTS SWITCH SIGNS BEFORE/AFTER
                    if lr >= descent_thresh:
                        loglr = log10(lr)
                        lr -= 10 ** (ceil(loglr) - 1)
                        lr = float(format(lr, ".{}f".format(str(2 + abs(floor(loglr))))))
                        print("LR", lr)
                    else:
                        return weights_record[np.argmin(vals)].tolist(), teams
                """
        except KeyboardInterrupt:
            return weights_record[np.argmin(vals)].tolist(), teams

if __name__ == "__main__":
    if not RESTORE:
        nows = [NOW]
    else:
        nows = {datetime(*[int(n) for n in i.split("/")[-1].split("-")[:-1]]) for i in glob("./Weekly/*.txt")}
    for i in nows:
        NOW = i
        rankings, teams = points_SGD(prop=False)
        #rankings = points_SGD([0 for _ in list(TEAMS)], TEAMS, prop=False)
        rankings = list(sorted(zip(teams.tolist(), rankings), key=lambda e: e[1], reverse=True))
        with open("./Weekly/{}-sgd_gravity.txt".format(str(NOW.date())), "w+") as doc:
            doc.write("RANKINGS\n" + rankings.__repr__().replace("), ", "),\n") + "\n\n")
        rankings, teams = points_SGD(prop=True)
        #rankings = points_SGD([0 for _ in list(TEAMS)], TEAMS, prop=True)
        rankings = list(sorted(zip(teams.tolist(), rankings), key=lambda e: e[1], reverse=True))
        with open("./Weekly/{}-sgd_gravity_prop.txt".format(str(NOW.date())), "w+") as doc:
            doc.write("RANKINGS\n" + rankings.__repr__().replace("), ", "),\n") + "\n\n")