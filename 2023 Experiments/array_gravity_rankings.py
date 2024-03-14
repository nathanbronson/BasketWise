from sportsipy.ncaab.teams import Teams
from copy import copy
from datetime import datetime
from scipy.optimize import basinhopping
from functools import partial
from itertools import product
from numpy import argmax
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from math import log10, ceil, floor
import numpy as np

GRADIENT_SWITCH_THRESH = .9
PROP = False
NOW = datetime(2023, 1, 23) #datetime.now()

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

"""
difs = np.array([])
games = np.array([])
pts = np.array([])
"""

def construct_difs(teams, prop=PROP):
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
        _difs.append(np.pad(a, (0, l - len(a)), constant_values=np.nan))
    difs = np.array(_difs)
    _games = []
    for a in games:
        _games.append(np.pad(a, ((0, l - len(a)), (0, 0)), constant_values=np.nan))
    games = np.array(_games)
    return difs, games

def points_array_element(points_dict, arr):
    if np.nan in arr:
        return np.nan
    else:
        try:
            return points_dict[arr[0]] - points_dict[arr[1]]
        except KeyError:
            return np.nan

def construct_points(points_dict, games):
    func = partial(points_array_element, points_dict)
    return np.apply_along_axis(func, 2, games)

def array_evaluate_points(difs, points_dict, games):
    return np.sum(np.nan_to_num(np.abs(difs - construct_points(points_dict, games))))

def array_points_gradient(difs, points_dict, games, team):
    points_0 = copy(points_dict)
    points_0[team.name] -= .5
    points_1 = copy(points_dict)
    points_1[team.name] += .5
    return array_evaluate_points(difs, points_1, games) - array_evaluate_points(difs, points_0, games)

def evaluate_team_rank(team, rankings):
    _rankings = copy(rankings)
    if type(_rankings[0]) not in [str]:
        str_idx = [type(i) for i in rankings[0]].index(str)
        _rankings = [i[str_idx] for i in _rankings]
    explained_results = [1 if (rankings.index(team.name) < rankings.index(game.opponent_name) and game.result == "Win") or (rankings.index(team.name) > rankings.index(game.opponent_name) and game.result != "Win") else 0 for game in filter(lambda e: e.datetime < NOW, team.schedule)]
    return (sum(explained_results), len(explained_results))

def evaluate_ranks(teams, rankings):
    results = (evaluate_team_rank(team, rankings) for team in teams)
    return sum((i[0] for i in results))/sum((n[1] for n in results))

def evaluate_team_points(team, points_dict, prop=PROP):
    point_diffs = []
    for game in filter(lambda e: e.datetime < NOW, team.schedule):
        try:
            point_diffs.append(abs(points_dict[team.name] + game.points_for/(1 if not prop else (game.points_against + game.points_for)) - points_dict[game.opponent_name] - game.points_against/(1 if not prop else (game.points_against + game.points_for))))
        except Exception as err:
            if str(err)[0] + str(err)[-1] != "''":
                pass#print(type(err), err)
    return sum(point_diffs), len(point_diffs)

def evaluate_points(teams, points, prop=PROP):
    points_dict = {t.name: i for t, i in zip(list(teams), points)}
    point_diffs = [evaluate_team_points(team, points_dict, prop=prop) for team in teams]
    return sum((i[0] for i in point_diffs))/sum((n[1] for n in point_diffs))

def points_gradient(teams, points, idx, prop=PROP):
    _points0 = copy(points)
    _points0[idx] -= .5
    _points1 = copy(points)
    _points1[idx] += .5
    return evaluate_points(teams, _points1, prop=prop) - evaluate_points(teams, _points0, prop=prop)

def points_SGD(ranks, teams, descent_thresh=1e-8, lr0=1000, gradient_switch_thresh=GRADIENT_SWITCH_THRESH, prop=PROP):
    weights = copy(ranks)
    difs, games = construct_difs(teams, prop=prop)
    weights_record = []
    ite = 0
    lr = lr0
    vals = []
    gradients = []
    with ProcessPoolExecutor(max_workers=7) as executor:
        while True:
            try:
                gradients.append([])
                ite += 1
                weights_record.append(weights)
                weights_f = copy(weights)
                grads = []
                points_dict = {t.name: w for t, w in zip(list(teams), weights_f)}
                for idx in list(teams):
                    grads.append(executor.submit(array_points_gradient, difs, points_dict, games, idx))
                for idx, _ in enumerate(weights):
                    print(idx, end="\r")
                    gradients[-1].append(grads[idx].result())
                    weights_f[idx] += -lr * gradients[-1][-1]
                weights = weights_f
                points_dict = {t.name: w for t, w in zip(list(teams), weights)}
                vals.append(array_evaluate_points(difs, points_dict, games))
                print(ite, vals[-1])
                if len(vals) >= 2:
                    if vals[-1] >= vals[-2]:
                        weights = weights_record[np.argmin(vals)]
                        if lr >= descent_thresh:
                            loglr = log10(lr)
                            lr -= 10 ** (ceil(loglr) - 1)
                            lr = float(format(lr, ".{}f".format(str(2 + abs(floor(loglr))))))
                            print("LR", lr)
                        else:
                            return weights
                        continue
                    u = np.unique(-1 * np.sign(gradients[-1]) == np.sign(gradients[-2]), return_counts=True)
                    if True in u[0] and {i : n for i, n in zip(*u)}[True]/sum(u[1]) >= gradient_switch_thresh:#if vals[-2] - vals[-1] <= descent_thresh * lr ** .5:#5 ** log10(lr) INSTEAD DETECT OVERSHOOT WHERE GRADIENTS SWITCH SIGNS BEFORE/AFTER
                        if lr >= descent_thresh:
                            loglr = log10(lr)
                            lr -= 10 ** (ceil(loglr) - 1)
                            lr = float(format(lr, ".{}f".format(str(2 + abs(floor(loglr))))))
                            print("LR", lr)
                        else:
                            return weights
            except KeyboardInterrupt:
                return weights
        """
        if vals[-1] <= descent_thresh:
            return weights
        lr = 10 ** ((20 - ite)/5)
        """

def exploration_rank(ranks, teams, pct_interval=.25, difference_threshold=1, evaluate_func=evaluate_ranks):
    weights = copy(ranks)
    master_interval = (1, len(weights))
    solution_intervals = [master_interval for _ in weights]
    dif = pct_interval * len(weights)
    while True:
        print(dif, end="\r")
        candidates = list(product(*[[solution_interval[0] + solution_interval[1] * i * pct_interval for i in range(int(1/pct_interval) + 1)] for solution_interval in solution_intervals]))
        optimal_solution = candidates[argmax([evaluate_func(teams, candidate) for candidate in candidates])]
        solution_intervals = [(center - dif, center + dif) for center in optimal_solution]
        solution_intervals = [(i[0] if i[0] >= master_interval[0] else master_interval[0], i[1] if i[1] <= master_interval[1] else master_interval[1]) for i in solution_intervals]
        pct_interval *= pct_interval
        dif = pct_interval * len(weights)
        if all((int(i[0] - i[1]) == 0 for i in solution_intervals)):
            break
    return weights

def basin_ranks(ranks, teams, point_func=evaluate_points):
    func = partial(point_func, teams)
    return basinhopping(func, ranks)

if __name__ == "__main__":
    TEAMS = Teams(configyear)
    for i in tqdm(list(TEAMS)):
        i.schedule
    #print(points_SGD([0 for _ in list(TEAMS)], TEAMS))
    #print(basin_ranks([0 for _ in list(TEAMS)], TEAMS))
    #print(exploration_rank([0 for _ in list(TEAMS)], TEAMS))
    rankings = points_SGD([0 for _ in list(TEAMS)], TEAMS, prop=False)
    rankings = list(sorted(zip([i.name for i in list(TEAMS)], rankings), key=lambda e: e[1], reverse=True))
    with open("./Weekly/{}-sgd_gravity.txt".format(str(NOW.date())), "w+") as doc:
        doc.write("RANKINGS\n" + rankings.__repr__().replace("), ", "),\n") + "\n\n")
    rankings = points_SGD([0 for _ in list(TEAMS)], TEAMS, prop=True)
    rankings = list(sorted(zip([i.name for i in list(TEAMS)], rankings), key=lambda e: e[1], reverse=True))
    with open("./Weekly/{}-sgd_gravity_prop.txt".format(str(NOW.date())), "w+") as doc:
        doc.write("RANKINGS\n" + rankings.__repr__().replace("), ", "),\n") + "\n\n")
    """
    ranks = [i.name for i in TEAMS]
    ranks = sample(ranks, k=len(ranks))
    ranks = exploration_optimization(ranks, TEAMS)
    print([(n + 1, i) for n, i in enumerate(ranks)])
    """