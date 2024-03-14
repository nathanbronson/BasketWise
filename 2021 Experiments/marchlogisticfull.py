# -*- coding: utf-8 -*-
"""MarchLogisticFull.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Tifciz6LZoWqwVnTfuWPSyPTV-mXL8Lx
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sportsreference.ncaab.teams import Teams
from os.path import isfile
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from threading import Thread

#game of form (perspective name, game object)

def get_team_data(team, year, teams=None):
  if teams is None:
    teams = Teams(year)
  team = [i for i in filter(lambda e: e.name == team, teams)][0]
  return (team.points/team.games_played, team.opp_points/team.games_played, team.strength_of_schedule)

def make_row(team1, team2, year, teams=None):
  if teams is None:
    teams = Teams(year)
  team1 = [i for i in filter(lambda e: e.name == team1, teams)][0]
  team2 = [i for i in filter(lambda e: e.name == team2, teams)][0]
  ref1 = find_other_perspective((team1.name, team1.schedule[-1]))
  ref2 = find_other_perspective((team2.name, team2.schedule[-1]))
  if (ref1.opponent_rank if not ref1.opponent_rank is None else ((get_team_data(ref1.opponent_name, 0, teams=teams)[2] * -1) if ref2.opponent_rank is None else 100)) > (ref2.opponent_rank if not ref2.opponent_rank is None else ((get_team_data(ref2.opponent_name, 0, teams=teams)[2] * -1)) if ref1.opponent_rank is None else 100):
    fav = team2
    und = team1
  else:
    fav = team1
    und = team2
  fav = get_team_data(fav.name, year, teams=teams)
  und = get_team_data(und.name, year, teams=teams)
  return [fav[0], und[0], fav[1], und[1], fav[2], und[2]]

def bracket_parse(path):
  bracket = pd.read_excel(path)
  bracketlist = []
  for i in bracket:
    bracketlist.append([i for i in filter(lambda e: type(e) == type(""), bracket[i])])
  return bracketlist

def game_in(game, all):
  for i in all:
    if game[0] == i[1].opponent_name and i[0] == game[1].opponent_name:
      return True
  return False

def build_tourney(year, teams=None):
  if teams is None:
    teams = Teams(year)
  tourney = []
  for i in teams:
    for n in filter(lambda e: e.type == "NCAA", i.schedule):
      if not game_in((i.name, n), tourney):
        tourney.append((i.name, n))
  return tourney

def build_row(game, teams=None):
  if teams is None:
    teams = Teams(game[1].datetime.year)
  other = find_other_perspective(game, teams=teams)
  if (game[1].opponent_rank if not game[1].opponent_rank is None else ((get_team_data(game[1].opponent_name, 0, teams=teams)[2] * -1) if other.opponent_rank is None else 100)) < (other.opponent_rank if not other.opponent_rank is None else ((get_team_data(other.opponent_name, 0, teams=teams)[2] * -1) if game[1].opponent_rank is None else 100)):
    fav = game[1].opponent_name
    favwin = 0 if game[1].result == "Win" else 1
    und = game[0]
  else:
    fav = game[0]
    favwin = 1 if game[1].result == "Win" else 0
    und = game[1].opponent_name
  row = [favwin]
  row.append([i for i in filter(lambda e: e.name == fav, teams)][0].points/[i for i in filter(lambda e: e.name == fav, teams)][0].games_played)
  row.append([i for i in filter(lambda e: e.name == und, teams)][0].points/[i for i in filter(lambda e: e.name == und, teams)][0].games_played)
  row.append([i for i in filter(lambda e: e.name == fav, teams)][0].opp_points/[i for i in filter(lambda e: e.name == fav, teams)][0].games_played)
  row.append([i for i in filter(lambda e: e.name == und, teams)][0].opp_points/[i for i in filter(lambda e: e.name == und, teams)][0].games_played)
  row.append([i for i in filter(lambda e: e.name == fav, teams)][0].strength_of_schedule)
  row.append([i for i in filter(lambda e: e.name == und, teams)][0].strength_of_schedule)
  return row

def build_row_known(fav, und, teams=None):
  if teams is None:
    teams = Teams(2021)
  row = []
  try:
    row.append([i for i in filter(lambda e: e.name == fav, teams)][0].points/[i for i in filter(lambda e: e.name == fav, teams)][0].games_played)
    row.append([i for i in filter(lambda e: e.name == und, teams)][0].points/[i for i in filter(lambda e: e.name == und, teams)][0].games_played)
    row.append([i for i in filter(lambda e: e.name == fav, teams)][0].opp_points/[i for i in filter(lambda e: e.name == fav, teams)][0].games_played)
    row.append([i for i in filter(lambda e: e.name == und, teams)][0].opp_points/[i for i in filter(lambda e: e.name == und, teams)][0].games_played)
    row.append([i for i in filter(lambda e: e.name == fav, teams)][0].strength_of_schedule)
    row.append([i for i in filter(lambda e: e.name == und, teams)][0].strength_of_schedule)
  except Exception as err:
    print(fav, und)
    raise err
  return row

def build_tourney_data(year):
  teams = Teams(year)
  tourney_data = []
  for i in build_tourney(year, teams=teams):
    tourney_data.append(build_row(i, teams=teams))
  df = pd.DataFrame()
  df["favwin01"] = [i[0] for i in tourney_data]
  df["ppgfav"] = [i[1] for i in tourney_data]
  df["ppgund"] = [i[2] for i in tourney_data]
  df["papgfav"] = [i[3] for i in tourney_data]
  df["papgund"] = [i[4] for i in tourney_data]
  df["sosfav"] = [i[5] for i in tourney_data]
  df["sosund"] = [i[6] for i in tourney_data]
  return df

def find_other_perspective(game, teams=None): #tournament only
  if teams is None:
    teams = Teams(game[1].datetime.year)
  for i in teams:
    if i.name == game[1].opponent_name:
      for n in filter(lambda e: e.type == "NCAA", i.schedule):
        if game_in((i.name, n), [game]):
          return n

def build_tourneys_data(years):
  all = pd.DataFrame()
  for i in years:
    all = pd.concat([all, build_tourney_data(i)])
  return all

#def fill_bracket(path):

if isfile("./tenyears.csv"):
  data = pd.read_csv("./tenyears.csv")
else:
  data = build_tourneys_data(range(2010, 2020))
data = sm.add_constant(data)
#data["_constant"] = [1 for i in range(len(data["favwin01"]))]
data.head()

Xtrain = data[["ppgfav", "ppgund", "papgfav", "papgund", "sosfav", "sosund", "const"]]
Ytrain = data[["favwin01"]]

log_reg = sm.Logit(Ytrain, Xtrain).fit()
print(log_reg.summary())

lin_reg = sm.OLS(Ytrain, Xtrain).fit()
print(lin_reg.summary())

secondround = pd.read_excel("secondroundimport.xlsx")
secondround["_constant"] = [1 for i in range(len(secondround["favwin01"]))]
predictions = list(map(round, log_reg.predict(secondround[["ppgfav", "ppgund", "papgfav", "papgund", "sosfav", "sosund", "_constant"]])))
print("Predictions:", predictions)
print("Actual:     ", [i for i in secondround["favwin01"]])

if not isfile("./tenyears.csv"):
  data.to_csv("./tenyears.csv")

def lin_mod(ppgfav, ppgund, papgfav, papgund, sosfav, sosund):
  #return .0208451*ppgfav + -.0111201*ppgund + -.0237251*papgfav + .0235944*papgund + .0264091*sosfav + -.0232987*sosund - .2007182
  return lin_reg.predict([ppgfav, ppgund, papgfav, papgund, sosfav, sosund, 1])[0]

def log_mod(ppgfav, ppgund, papgfav, papgund, sosfav, sosund):
  return log_reg.predict([ppgfav, ppgund, papgfav, papgund, sosfav, sosund, 1])[0]

def round_calc(cut, stat, callb=lin_mod):
  return 1 if callb(*stat) > cut else 0

def cor(act, cut, stat, callb=lin_mod):
  return 1 if act == round_calc(cut, stat, callb=callb) else 0

def cors(acts, cut, stats, callb=lin_mod):
  return [cor(acts[i], cut, stats[i], callb=callb) for i in range(len(acts))]

def dfcors(df, cut, callb=lin_mod):
  acts = df["favwin01"].values
  stats = df[["ppgfav", "ppgund", "papgfav", "papgund", "sosfav", "sosund"]].values
  return cors(acts, cut, stats, callb=callb)

accs = [i/10000 for i in range(10000)]
lintotals = []
logtotals = []
for i in accs:
  lintotals.append(sum(dfcors(data, i)))
  logtotals.append(sum(dfcors(data, i, callb=log_mod)))
lintotals = [i/667 for i in lintotals]
logtotals = [i/667 for i in logtotals]

plt.plot(accs, lintotals, color="red")
plt.plot(accs, logtotals, color="blue")

def add_pg(df):
  df["points_per_game0"] = [df["points0"].values[i]/df["games_played0"].values[i] for i in range(len(df["points0"].values))]
  df["points_per_game1"] = [df["points1"].values[i]/df["games_played1"].values[i] for i in range(len(df["points0"].values))]
  df["points_allowed_per_game0"] = [df["opp_points0"].values[i]/df["games_played0"].values[i] for i in range(len(df["points0"].values))]
  df["points_allowed_per_game1"] = [df["opp_points1"].values[i]/df["games_played1"].values[i] for i in range(len(df["points0"].values))]
  return df

def get_fields(year, teams=None):
  if teams is None:
    teams = Teams(year)
  fields = []
  r = 0
  for n in teams:
    r = n
    break
  for i in r.dataframe:
    try:
      int(r.dataframe[i][0])
      fields.append(i)
    except:
      pass
  return fields

def get_col_data(cols, suf, team, year, teams=None):
  if teams is None:
    teams = Teams(year)
  df = None
  for i in filter(lambda e: e.name == team, teams):
    df = i.dataframe[cols]
    df.columns = [str(n) + str(suf) for n in df.columns]
    return df

def splice(df1, df2):
  n = 0
  for _ in df1.iterrows():
    n += 1
  df1.index = [str(i) for i in range(n)]
  df2.index = [str(i) for i in range(n)]
  return pd.concat([df1, df2], axis=1)

def build_full_row(game, teams=None):
  if teams is None:
    teams = Teams(game[1].datetime.year)
  other = find_other_perspective(game, teams=teams)
  if (game[1].opponent_rank if not game[1].opponent_rank is None else ((get_team_data(game[1].opponent_name, 0, teams=teams)[2] * -1) if other.opponent_rank is None else 100)) < (other.opponent_rank if not other.opponent_rank is None else ((get_team_data(other.opponent_name, 0, teams=teams)[2] * -1) if game[1].opponent_rank is None else 100)):
    fav = game[1].opponent_name
    favwin = 0 if game[1].result == "Win" else 1
    und = game[0]
  else:
    fav = game[0]
    favwin = 1 if game[1].result == "Win" else 0
    und = game[1].opponent_name
  row = splice(get_col_data(get_fields(2021, teams=teams), "0", und, 2021, teams=teams), get_col_data(get_fields(2021, teams=teams), "1", fav, 2021, teams=teams))
  row["favwin01"] = favwin
  return row

def build_full_tourney_data(year):
  teams = Teams(year)
  tourney_data = pd.DataFrame()
  n = 0
  for i in build_tourney(year, teams=teams):
    new = build_full_row(i, teams=teams)
    new.index = [n]
    n += 1
    tourney_data = pd.concat([tourney_data, new], axis=0)
  return tourney_data

def build_full_tourneys_data(years):
  all = pd.DataFrame()
  for i in years:
    all = pd.concat([all, build_full_tourney_data(i)], axis=0)
  return all

if isfile("./fulltenyears.csv"):
  fulldata = pd.read_csv("./fulltenyears.csv")
  #fulldata = fulldata.drop("")
else:
  fulldata = add_pg(build_full_tourneys_data(range(2010, 2020)))
fulldata["_constant"] = [1 for i in range(len(fulldata["favwin01"]))]
#fulldata = sm.add_constant(fulldata)
fulldata.head()

if not isfile("./fulltenyears.csv"):
  fulldata.to_csv("./fulltenyears.csv")

fullXtrain = fulldata.drop("favwin01", axis=1)
try:
  fullXtrain = fullXtrain.drop("Unnamed: 0", axis=1)
except Exception as err:
  print(err)
while True:
  try:
    fullXtrain = fullXtrain.drop("_constant")
  except:
    break
fullYtrain = fulldata[["favwin01"]]
fullXtrain.head()

rfe_full_log_reg = LogisticRegression(max_iter=1000000000000, verbose=False)
rfe = RFE(rfe_full_log_reg, 19, verbose=False).fit(fullXtrain, fullYtrain.values.ravel())
sup = rfe.support_
print(sup)
ind = [i for i in filter(lambda e: sup[e], [n for n in range(len(sup))])]
while True:
  try:
    fullXtrain = fullXtrain.drop("_constant")
  except:
    break
keys = [fullXtrain.columns.values[i] for i in ind] + ["_constant"]
fullXtrain["_constant"] = [1 for i in range(len(fullXtrain[fullXtrain.columns.values[0]]))]
fullXtrain = fullXtrain[keys]
fullXtrain.head(700)

assert len([i for i in filter(lambda e: not e == 1 and not e == 0, [n for n in fullYtrain["favwin01"].values])]) == 0
full_log_reg = sm.Logit(fullYtrain, fullXtrain).fit()
print(full_log_reg.summary())

full_lin_reg = sm.OLS(fullYtrain, fullXtrain).fit()
print(full_lin_reg.summary())

def full_lin_mod(allstat):
  #return .0208451*ppgfav + -.0111201*ppgund + -.0237251*papgfav + .0235944*papgund + .0264091*sosfav + -.0232987*sosund - .2007182
  return full_lin_reg.predict(allstat)[0]

def full_log_mod(allstat):
  return full_log_reg.predict(allstat)[0]

def full_round_calc(cut, stat, callb=full_lin_mod):
  return 1 if callb(stat) > cut else 0

def fullcor(act, cut, stat, callb=full_lin_mod):
  return 1 if act == full_round_calc(cut, stat, callb=callb) else 0

def fullcors(acts, cut, stats, callb=full_lin_mod):
  return [fullcor(acts[i], cut, stats[i], callb=callb) for i in range(len(acts))]

def fulldfcors(df, ydf, cut, callb=full_lin_mod, teams=None):
  if teams is None:
    teams = Teams(2021)
  acts = ydf.values
  stats = df.values
  return fullcors(acts, cut, stats, callb=callb)

accs = [i/10000 for i in range(10000)]
fulllintotals = []
fulllogtotals = []
teams = Teams(2021)
for i in accs:
  fulllintotals.append(sum(fulldfcors(fullXtrain, fullYtrain, i, teams=teams)))
  fulllogtotals.append(sum(fulldfcors(fullXtrain, fullYtrain, i, callb=full_log_mod, teams=teams)))
fulllintotals = [i/667 for i in fulllintotals]
fulllogtotals = [i/667 for i in fulllogtotals]

plt.clf()
plt.plot(accs, fulllintotals, color="red")
plt.plot(accs, fulllogtotals, color="blue")

def build_matchup(fav, und, teams=None):
  if teams is None:
    teams = Teams(2021)
  fav, und = get_fav(fav, und, teams=teams)
  row = splice(get_col_data(get_fields(2021, teams=teams), "0", und[1], 2021, teams=teams), get_col_data(get_fields(2021, teams=teams), "1", fav[1], 2021, teams=teams))
  row["_constant"] = [1]
  row = add_pg(row)
  row = row[keys].values
  return row

def get_fav(fav, und, teams=None):
  if fav[0] > und[0]:
    return und, fav
  else:
    return fav, und

def populate_bracket(r1, predict, data, year=2021, teams=None):
  if teams is None:
    teams = Teams(year)
  rounds = [r1]
  remaining = r1
  while len(remaining) > 1:
    victors = []
    for i in range(int(len(remaining)/2)):
      victors.append(remaining[i * 2] if round(predict(data(remaining[i * 2], remaining[(i * 2) + 1], teams=teams))) == 1 else remaining[(i * 2) + 1])
    remaining = victors
    rounds.append(victors)
  return rounds

def wrap_build(fav, und, teams=None):
  if teams is None:
    teams = Teams(2021)
  return build_row_known(fav[1], und[1], teams=teams)

def wrap_lin(stat):
  return lin_mod(*stat)

def wrap_log(stat):
  return log_mod(*stat)

def load_bracket(path):
  with open(path, "r") as doc:
    l = eval(doc.read())
  return l

def get_estimates(year, path):
  teams = Teams(year)
  print("lin_mod:", populate_bracket(load_bracket(path), wrap_lin, wrap_build, teams=teams))
  print("log_mod:", populate_bracket(load_bracket(path), wrap_log, wrap_build, teams=teams))
  print("full_lin_mod:", populate_bracket(load_bracket(path), full_lin_mod, build_matchup, teams=teams))
  print("full_log_mod:", populate_bracket(load_bracket(path), full_log_mod, build_matchup, teams=teams))

get_estimates(2021, "bracket.txt")