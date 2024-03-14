import sportsdataverse
from sportsipy.ncaab.teams import Teams
import tensorflow as tf
import numpy as np
from dataset import p_len, s_len, p_types, s_types, lk
from optimized_winformer import OUTPUT_D, D_MODEL, DFF, WinFormer, CHECKPOINT_PATH
from game_sequence import match_team, valid_keys, pg_keys, base_seconds
from json import load
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import ceil
from dataset import MAX_LEN
from tqdm import tqdm

with tf.device("/cpu:0"):
    class CGame(object):
        def __init__(self, game_id, home_id, away_id, play_types, committing_teams, scores, times, stats):
            self.stats = stats
            self.game_id = game_id
            self.home_id = int(home_id)
            self.away_id = int(away_id)
            self.times = times
            #self.home_masks = committing_teams == self.home_id
            self.home_masks = np.vectorize(lambda e: {self.away_id: -1, self.home_id: 1, str(self.away_id): -1, str(self.home_id): 1, None: 0}[e])(committing_teams)
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
            #might need to get game result here
        
        def __str__(self):
            return " ".join([str(self.away_id), "at", str(self.home_id), "on", self.game_date.strftime("%a, %b %-d, %Y")])
        
        def home_perspective(self):
            #arrange stats, arrange scores, mask_play_types
            return self.home_masked_plays + self.none_masked_plays, self.home_masked_scores, self.times, self.stats
        
        def away_perspective(self):
            return -1 * self.home_masked_plays + self.none_masked_plays, -1 * self.home_masked_scores, self.times, self.stats[::-1]

    class WinPercentage(object):
        def __init__(self, game_id, refresh_sec=2, year=2023):
            self.home_win_prob = []
            self.refresh_sec = refresh_sec
            self.game_id = game_id
            model = WinFormer(
                d_model=D_MODEL,
                dff=DFF,
                output_d=OUTPUT_D,
                num_score_types=s_types,
                num_play_types=p_types,
                num_stats_keys=s_len,
                dropout_rate=0.0
            )
            model.game_former.positional_encoder.embedding.set_embeddings(l=lk)
            model.load_weights(CHECKPOINT_PATH)
            self.model = model
            teams = Teams(year)
            pbp = sportsdataverse.mbb.mbb_pbp.espn_mbb_pbp(self.game_id)
            svstat = sportsdataverse.mbb.mbb_loaders.load_mbb_team_boxscore([year])
            self.home_id = int(pbp["teamInfo"]["home"]["id"])
            for team in list(teams):
                m = match_team(team, svstat)
                if m is None:
                    continue
                else:
                    m = int(m)
                if m == self.home_id:
                    self.home_team = team
                    break
            self.away_id = int(pbp["teamInfo"]["away"]["id"])
            for team in list(teams):
                m = match_team(team, svstat)
                if m is None:
                    continue
                else:
                    m = int(m)
                if m == self.away_id:
                    self.away_team = team
                    break
            for column in pg_keys:
                self.home_team.dataframe[column] = self.home_team.dataframe[column].values/self.home_team.dataframe["games_played"].values
                self.away_team.dataframe[column] = self.away_team.dataframe[column].values/self.away_team.dataframe["games_played"].values
            with open("./windata_stats.json", "r") as doc:
                stat = load(doc)
            with open("./windata_lookups.json", "r") as doc:
                self.lookups = load(doc)
            self.play_lookup = np.vectorize(lambda e: {int(k): int(v) for k, v in self.lookups["plays"].items()}[e])
            self.score_lookup = np.vectorize(lambda e: {int(k): int(v) for k, v in self.lookups["scores"].items()}[e])
            for column in valid_keys:
                self.home_team.dataframe[column] = (self.home_team.dataframe[column].values - stat["means"][column])/stat["stds"][column]
                self.away_team.dataframe[column] = (self.away_team.dataframe[column].values - stat["means"][column])/stat["stds"][column]
            self.home_data = self.home_team.dataframe[valid_keys].values.flatten()
            self.away_data = self.away_team.dataframe[valid_keys].values.flatten()
        
        def get_data(self):
            pbp = sportsdataverse.mbb.mbb_pbp.espn_mbb_pbp(self.game_id)
            committing_teams = np.array([p["team.id"] for p in pbp["plays"]])#.astype("int32")
            plays = np.array([p["type.id"] for p in pbp["plays"]]).astype("int32")
            scores = np.array([p["scoreValue"] for p in pbp["plays"]]).astype("int32")
            period = np.array([p["period"] for p in pbp["plays"]]).astype("int32")
            minutes = np.array([p["clock.minutes"] for p in pbp["plays"]]).astype("int32")
            seconds = np.array([p["clock.seconds"] for p in pbp["plays"]]).astype("int32")
            times = np.vectorize(base_seconds)(period) - 60 * minutes - seconds
            game = CGame(self.game_id, self.home_id, self.away_id, plays, committing_teams, scores, times, (self.home_data, self.away_data))
            return game.home_perspective(), game.away_perspective()
        
        def main_ret(self):
            self.home_win_prob = []
            while True:
                home, away = self.get_data()
                if home[2][-1] > self.home_win_prob[-1][0]:
                    self.home_win_prob.append((home[2][-1], float(np.mean(np.concatenate([self.model([home]), 1 - self.model([away])])))))
                yield self.home_win_prob[-1][1]
                sleep(self.refresh_sec)
        
        def main_set(self):
            home, away = self.get_data()
            dif = len(home[2]) - len(self.home_win_prob)
            #if home[2][-1] > self.home_win_prob[-1][0]:
            if dif > 0:
                for i in tqdm(list(range(dif))[::-1]):
                    altered_home = (np.expand_dims(np.pad(self.play_lookup(home[0][:-1 - i]) if abs(-1 - i) < len(home[0]) else [], (0, MAX_LEN), "constant").flatten()[:MAX_LEN], 0), np.expand_dims(np.pad(self.score_lookup(home[1][:-1 - i]) if abs(-1 - i) < len(home[1]) else [], (0, MAX_LEN), "constant").flatten()[:MAX_LEN], 0), np.expand_dims(np.pad(home[2][:-1 - i], (0, MAX_LEN), "constant").flatten()[:MAX_LEN], 0), (np.expand_dims(home[3][0], 0), np.expand_dims(home[3][1], 0)))
                    altered_away = (np.expand_dims(np.pad(self.play_lookup(away[0][:-1 - i]) if abs(-1 - i) < len(away[0]) else [], (0, MAX_LEN), "constant").flatten()[:MAX_LEN], 0), np.expand_dims(np.pad(self.score_lookup(away[1][:-1 - i]) if abs(-1 - i) < len(away[1]) else [], (0, MAX_LEN), "constant").flatten()[:MAX_LEN], 0), np.expand_dims(np.pad(away[2][:-1 - i], (0, MAX_LEN), "constant").flatten()[:MAX_LEN], 0), (np.expand_dims(away[3][0], 0), np.expand_dims(away[3][1], 0)))
                    self.home_win_prob.append((home[2][-1 - i], float(np.mean(np.concatenate([self.model(altered_home), 1 - self.model(altered_away)])))))
            return self.home_win_prob
        
        def get_tp(self):
            return list(zip(*self.home_win_prob))
        
        def main_graph(self):
            fig = plt.figure()
            ax1 = fig.add_subplot(1,1,1)
            plt.ylim(0, 1)
            plt.xlim(0, 2400)
            def animate(i):
                data = self.main_set()
                xar = [i[0] for i in data]
                yar = [i[1] for i in data]
                if xar[-1] > 2400:
                    plt.ylim(0, 1)
                    plt.xlim(0, 2400 + 300 * ceil((xar[-1] - 2400)/300))
                ax1.clear()
                ax1.plot(xar, yar)
            ani = animation.FuncAnimation(fig, animate, interval=self.refresh_sec * 1000)
            plt.show()

    if __name__ == "__main__":
        w = WinPercentage(401514242)
        w.main_graph()