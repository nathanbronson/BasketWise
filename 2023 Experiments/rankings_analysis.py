from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy
from statistics import mean
from datetime import datetime

VERTICAL = False

def refine(s):
    return s.replace("'St. Thomas'", "'St. Thomas (MN)'")

def date_decompose(s):
    return datetime.strptime("-".join(s.split("/")[-1].split(".")[0].split("-")[:-1]), "%Y-%m-%d")

files = sorted(glob("./Weekly/*.txt"), key=date_decompose)
matchups = list(filter(lambda e: "matchups" in e, files))
tournament = list(filter(lambda e: "tournament" in e, files))

matchups_data = []
for i in matchups:
    with open(i, "r+") as doc:
        r = doc.read()
        matchups_data.append(eval(refine(r[r.index("RANKINGS\n"):].replace("RANKINGS\n", ""))))
tournament_data = []
for i in tournament:
    with open(i, "r+") as doc:
        r = doc.read()
        tournament_data.append(eval(refine(r[r.index("RANKINGS\n"):].replace("RANKINGS\n", ""))))

def move_val(initial, final, value, default=0, t=False):
    try:
        return final[value] - initial[value]
    except KeyError as err:
        if not t:
            raise err
        try:
            return final[value]
        except KeyError:
            try:
                return 26 - initial[value]
            except KeyError:
                return default

def average_move_size(l, only_move=True, key=-1, top=25, ret_list=False, ret_time_sequence=False, ret_total=False, move=move_val):
    _l = [sorted(i, key=lambda e: e[0]) for i in l]
    rounds = [{n[1]: n[key] for n in i} for i in _l]
    movement = []
    for i0, i1 in zip(rounds[:-1], rounds[1:]):
        movement.append({i: move(i0, i1, i) for i in i1})
    if ret_list:
        return movement
    movement = [mean([n for g, n in i.items() if (n > 0 or not only_move)]) for i in movement]
    if ret_time_sequence:
        return movement
    if ret_total:
        return mean(movement)

if __name__ == "__main__":
    print("MATCHUP METRIC")
    print("EXCLUDING", average_move_size(matchups_data, ret_time_sequence=True, key=-1), average_move_size(matchups_data, ret_total=True, key=-1))
    print("INCLUDING", average_move_size(matchups_data, ret_time_sequence=True, only_move=False, key=-1), average_move_size(matchups_data, ret_total=True, only_move=False, key=-1))
    print("RANKINGS")
    print("EXCLUDING", average_move_size(matchups_data, ret_time_sequence=True, key=0), average_move_size(matchups_data, ret_total=True, key=0))
    print("INCLUDING", average_move_size(matchups_data, ret_time_sequence=True, only_move=False, key=0), average_move_size(matchups_data, ret_total=True, only_move=False, key=0))
    for idx in range(len(matchups)):
        ax = plt.subplot(*(len(matchups), 2)[::1 if VERTICAL else -1], (idx * 2 + 1 if VERTICAL else idx + 1))
        ax.hist([i[-1] for i in matchups_data[idx][:25]], bins=int(9), edgecolor="black")
        ax = plt.subplot(*(len(matchups), 2)[::1 if VERTICAL else -1], (idx * 2 + 2 if VERTICAL else idx + 1 + len(matchups)))
        ax.hist([i[-1] for i in tournament_data[idx][:25]], bins=int(9), edgecolor="black")

    plt.show()