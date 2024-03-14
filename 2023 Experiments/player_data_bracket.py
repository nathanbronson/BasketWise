from sportsipy.ncaab.teams import Teams
from player_data_creator import keys
from player_dataset import roster_len, stat_len, sort_idx, _sort
import numpy as np
from playerformer import PlayerFormer, D_MODEL, DFF, dropout_rate, CHECKPOINT_PATH
from functools import partial
from json import load

CONFIG_YEAR = 2023
TEAMS = Teams(CONFIG_YEAR)
BRACKET = [(1, "Alabama"), (16, "Texas A&M-Corpus Christi"), (8, "Maryland"), (9, "West Virginia"), (5, "San Diego State"), (12, "College of Charleston"), (4, "Virginia"), (13, "Furman"), (6, "Creighton"), (11, "NC State"), (3, "Baylor"), (14, "UC Santa Barbara"), (7, "Missouri"), (10, "Utah State"), (2, "Arizona"), (15, "Princeton"), (1, "Purdue"), (16, "Texas Southern"), (8, "Memphis"), (9, "Florida Atlantic"), (5, "Duke"), (12, "Oral Roberts"), (4, "Tennessee"), (13, "Louisiana"), (6, "Kentucky"), (11, "Providence"), (3, "Kansas State"), (14, "Montana State"), (7, "Michigan State"), (10, "Southern California"), (2, "Marquette"), (15, "Vermont"), (1, "Houston"), (16, "Northern Kentucky"), (8, "Iowa"), (9, "Auburn"), (5, "Miami (FL)"), (12, "Drake"), (4, "Indiana"), (13, "Kent State"), (6, "Iowa State"), (11, "Mississippi State"), (3, "Xavier"), (14, "Kennesaw State"), (7, "Texas A&M"), (10, "Penn State"), (2, "Texas"), (15, "Colgate"), (1, "Kansas"), (16, "Howard"), (8, "Arkansas"), (9, "Illinois"), (5, "Saint Mary's (CA)"), (12, "Virginia Commonwealth"), (4, "Connecticut"), (13, "Iona"), (6, "TCU"), (11, "Nevada"), (3, "Gonzaga"), (14, "Grand Canyon"), (7,  "Northwestern"), (10, "Boise State"), (2, "UCLA"), (15, "UNC Asheville")]
BATCH_SIZE = 512
pad_val = [0 for _ in range(len(keys))]

with open("./playerdata_stats.json", "r") as doc:
    st = load(doc)

means = st["means"]
stds = st["stds"]

model = PlayerFormer(
    d_model=D_MODEL,
    dff=DFF,
    num_stats_keys=stat_len,
    dropout_rate=dropout_rate
)
model.load_weights(CHECKPOINT_PATH)

def build_data(team1, team2):
    t1 = list(filter(lambda e: e.name == team1, list(TEAMS)))[0].roster.all_players
    t2 = list(filter(lambda e: e.name == team2, list(TEAMS)))[0].roster.all_players
    d1 = _sort(((np.array([[v[key] for key in keys] for _, v in t1.items()]) - means)/stds).tolist())
    d2 = _sort(((np.array([[v[key] for key in keys] for _, v in t2.items()]) - means)/stds).tolist())
    return d1 + [pad_val for _ in range(roster_len - len(d1))], d2 + [pad_val for _ in range(roster_len - len(d2))]

def evaluate(data1, data2, sample=True):
    if sample:
        d1 = np.array([np.random.permutation(i) for i in np.repeat(np.expand_dims(data1, 0), BATCH_SIZE, axis=0).tolist()])
        d2 = np.array([np.random.permutation(i) for i in np.repeat(np.expand_dims(data2, 0), BATCH_SIZE, axis=0).tolist()])
        pred1 = model((d1, d2)).numpy()
        pred2 = model((d2, d1)).numpy()
        pred = (1 + pred1 - pred2)/2
        return np.mean(pred)#probability that team 1 wins
    else:
        return model((np.expand_dims(data1, 0), np.expand_dims(data2, 0))).numpy()

def run(team1, team2, sample=True):
    return evaluate(*build_data(team1, team2), sample=sample)

def pick(team1, team2):
    if type(team1) is tuple and type(team2) is tuple:
        return (team1, team2)[np.round(run(team1[1], team2[1]))]
    else:
        return (team1, team2)[np.round(run(team1, team2))]

def populate_bracket(r1, predict=partial(run, sample=False)):
    rounds = [r1]
    remaining = r1
    while len(remaining) > 1:
        victors = []
        for i in range(int(len(remaining)/2)):
            result = remaining[i * 2] if np.round(predict(remaining[i * 2][1], remaining[(i * 2) + 1][1])) == 1 else remaining[(i * 2) + 1]
            victors.append(result)
        remaining = victors
        rounds.append(victors)
    return rounds

if __name__ == "__main__":
    #pass
    #print(run("Kansas", "TCU"))
    #print(run("TCU", "Kansas"))
    #print(run("Kansas", "TCU", sample=True))
    #print(run("TCU", "Kansas", sample=True))
    with open("./bracket0.txt", "r") as doc:
        print(populate_bracket(eval(doc.read())))