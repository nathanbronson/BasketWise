from sportsipy.ncaab.teams import Teams
from copy import copy
from datetime import datetime
from tqdm import tqdm
from math import log10, ceil, floor
import numpy as np
from pickle import dump, load
import os
from glob import glob

GRADIENT_SWITCH_THRESH = .9
PROP = False
NOW = datetime.now()
LOAD_TEAMS = False
RESTORE = False
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

def construct_difs(teams, prop=PROP, normalize=NORMALIZE_DIFS):
    difs = []
    lens = []
    games = []
    for team in list(teams):
        g = list(filter(lambda c: c.points_for is not None, filter(lambda e: e.datetime < NOW, team.schedule)))
        difs.append(np.array([(game.points_for - game.points_against)/((game.points_for + game.points_against) if prop else 1) for game in g], dtype=np.float32))
        lens.append(len(difs[-1]))
        games.append(np.array([[team.name, game.opponent_name] for game in g]))
    l = max(lens)
    _difs = []
    for a in difs:
        _difs.append(np.pad(a, (0, l - len(a) + 1), constant_values=np.nan))
    difs = np.array(_difs)
    _games = []
    for a in games:
        _games.append(np.pad(a, ((0, l - len(a) + 1), (0, 0)), constant_values=np.nan))
    #_games.append(np.array([[np.nan, np.nan] for _ in range(l)]))
    games = np.array(_games)
    ndifs = np.nan_to_num(difs)
    if normalize:
        return (difs - np.mean(ndifs))/np.std(ndifs), games
    else:
        return difs, games

def points_SGD(ranks, teams, descent_thresh=1e-12, lr0=1000, gradient_switch_thresh=GRADIENT_SWITCH_THRESH, prop=PROP):
    weights = np.array(copy(ranks))
    l = len(weights)
    difs, games = construct_difs(teams, prop=prop)
    t = [i.name for i in list(teams)]
    points_indices_0 = []
    points_indices_1 = []
    for team in games:
        points_indices_0.append([])
        points_indices_1.append([])
        for game in team:
            try:
                points_indices_0[-1].append(t.index(game[0]))
            except:
                points_indices_0[-1].append(-1)
            try:
                points_indices_1[-1].append(t.index(game[1]))
            except:
                points_indices_1[-1].append(-1)
    points_indices_0 = np.tile(np.array(points_indices_0, dtype=np.int_), (l, 1, 1))
    points_indices_1 = np.tile(np.array(points_indices_1, dtype=np.int_), (l, 1, 1))
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
                        return weights_record[np.argmin(vals)].tolist()
                    continue
                u = np.unique(-1 * np.sign(gradients[-1]) == np.sign(gradients[-2]), return_counts=True)
                if True in u[0] and {i : n for i, n in zip(*u)}[True]/sum(u[1]) >= gradient_switch_thresh:#if vals[-2] - vals[-1] <= descent_thresh * lr ** .5:#5 ** log10(lr) INSTEAD DETECT OVERSHOOT WHERE GRADIENTS SWITCH SIGNS BEFORE/AFTER
                    if lr >= descent_thresh:
                        loglr = log10(lr)
                        lr -= 10 ** (ceil(loglr) - 1)
                        lr = float(format(lr, ".{}f".format(str(2 + abs(floor(loglr))))))
                        print("LR", lr)
                    else:
                        return weights_record[np.argmin(vals)].tolist()
        except KeyboardInterrupt:
            return weights_record[np.argmin(vals)].tolist()

if __name__ == "__main__":
    if not RESTORE:
        nows = [NOW]
    else:
        nows = {datetime(*[int(n) for n in i.split("/")[-1].split("-")[:-1]]) for i in glob("./Weekly/*.txt")}
    if LOAD_TEAMS and os.path.isfile("./teams_obj.pkl"):
        with open("./teams_obj.pkl", "rb") as doc:
            TEAMS = load(doc)
    else:
        TEAMS = Teams(configyear)
        for i in tqdm(list(TEAMS)):
            i.schedule
    if LOAD_TEAMS:
        with open("./teams_obj.pkl", "wb") as doc:
            dump(TEAMS, doc)
    for i in tqdm(list(TEAMS)):
        i.schedule
    #print(points_SGD([0 for _ in list(TEAMS)], TEAMS))
    #print(basin_ranks([0 for _ in list(TEAMS)], TEAMS))
    #print(exploration_rank([0 for _ in list(TEAMS)], TEAMS))
    for i in nows:
        NOW = i
        rankings = points_SGD(10 * np.random.rand(len(list(TEAMS))), TEAMS, prop=False)
        #rankings = points_SGD([0 for _ in list(TEAMS)], TEAMS, prop=False)
        rankings = list(sorted(zip([i.name for i in list(TEAMS)], rankings), key=lambda e: e[1], reverse=True))
        with open("./Weekly/{}-sgd_gravity.txt".format(str(NOW.date())), "w+") as doc:
            doc.write("RANKINGS\n" + rankings.__repr__().replace("), ", "),\n") + "\n\n")
        rankings = points_SGD(np.random.rand(len(list(TEAMS))), TEAMS, prop=True)
        #rankings = points_SGD([0 for _ in list(TEAMS)], TEAMS, prop=True)
        rankings = list(sorted(zip([i.name for i in list(TEAMS)], rankings), key=lambda e: e[1], reverse=True))
        with open("./Weekly/{}-sgd_gravity_prop.txt".format(str(NOW.date())), "w+") as doc:
            doc.write("RANKINGS\n" + rankings.__repr__().replace("), ", "),\n") + "\n\n")
    """
    ranks = [i.name for i in TEAMS]
    ranks = sample(ranks, k=len(ranks))
    ranks = exploration_optimization(ranks, TEAMS)
    print([(n + 1, i) for n, i in enumerate(ranks)])
    """