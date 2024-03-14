import matplotlib.pyplot as plt
from sportsipy.ncaab.teams import Teams
from tqdm import tqdm
from pickle import load, dump
import os

LOAD_TEAMS = False
configyear = 2023

if __name__ == "__main__":
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
    candidates = []
    AB = []#A-B
    AC = []#(A-C)-(B-C)
    BC = []
    for i in tqdm(list(TEAMS)):
        i.schedule
    for team in tqdm(list(TEAMS)):
        sched = [i.opponent_name for i in team.schedule]
        for primary in sched:
            for secondary in sched:
                try:
                    if primary in [i.opponent_name for i in list(filter(lambda e: e.name == secondary, list(TEAMS)))[0].schedule]:
                        candidates.append((team.name, primary, secondary))
                except Exception as err:
                    pass
    for candidate in tqdm(candidates):
        try:
            a = list(filter(lambda e: e.name == candidate[0], list(TEAMS)))[0]
            b = list(filter(lambda e: e.name == candidate[1], list(TEAMS)))[0]
            c = list(filter(lambda e: e.name == candidate[2], list(TEAMS)))[0]
        except Exception as err:
            continue
        ab = list(filter(lambda e: e.opponent_name == b.name, a.schedule))[0]
        ac = list(filter(lambda e: e.opponent_name == c.name, a.schedule))[0]
        bc = list(filter(lambda e: e.opponent_name == c.name, b.schedule))[0]
        if None in [ab.points_for, ab.points_against, ac.points_for, ac.points_against, bc.points_for, bc.points_against]:
            continue
        AB.append(ab.points_for - ab.points_against)
        AC.append(ac.points_for - ac.points_against)
        BC.append(bc.points_for - bc.points_against)
    with open("./linearity_output.txt", "w+") as doc:
        doc.write("CANDIDATES\n" + candidates.__repr__() + "\nAB\n" + AB.__repr__() + "\nAC\n" + AC.__repr__() + "\nBC\n" + BC.__repr__() + "\n")
    difs = x = [ac - bc for ac, bc in zip(AC, BC)]
    spans = y = AB
    plt.scatter(x, y)
    plt.show()