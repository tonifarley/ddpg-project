"""Environment configurations

Defines action space, state space, initial state

"""

import sys
import pandas as pd
import numpy as np
from collections import namedtuple
from scipy import stats

class RANGE:
    actions = range(0, 3)
    state = [
        ['d', 0, 300],
        ['lat', 0, 90],
        ['lon', -180, 0],
        ['soc', 0, 100],
        ['day', 0, 6]
    ]

    def __init__(self, cid, df):
        self.id = cid
        self.df = self.preprocess(df)
        self.goal = self.df.rr
        self.reset()

    def reset(self):
        self.attempt = [(0, self.df.iloc[0].rr, self.roc)]

    def preprocess(self, df):
        delta = df.iloc[[0, -1]].diff()
        dsoc = delta.iloc[-1].soc
        dd = delta.iloc[-1].d
        if dsoc >= 0:
            raise ValueError(f'dsoc = {dsoc} for {self.id}')
        elif dd <= 0:
            raise ValueError(f'dd = {dd} for {self.id}')
        else:
            self.roc = abs(dd/dsoc)
            df['rr'] = df.soc * self.roc
        return df

    def obs(self, index=0):
        row = self.df.iloc[index]
        return [row.d, row.lat, row.lon, row.soc, row.day]

    def step(self, action, start, stop):
        seg = self.df[start:stop]
        roc = action[0]

        obs = seg.iloc[-1]
        p = obs.soc * roc
        a = obs.rr
        d = abs(p-a)
        reward = 1 - min(1, d/100)

        self.attempt.append((obs.name, p, roc))
        obs = self.obs(obs.name)
        done = stop >= len(self.df)
        return obs, reward, done, {'a': a, 'p': p, 'pr': roc}

class CHARGE:
    actions = range(1, 20)
    state = [
        ['t', 0, np.Inf],
        ['soc', 0, 100],
        ['day', 0, 6],
        ['energy', -100, 300],
        ['voltage', 0, 700],
        ['ambient_temp', 0, 100]
    ]

    def __init__(self, cid, df):
        self.id = cid
        self.df = self.preprocess(df)
        self.goal = self.df.ct
        self.reset()

    def reset(self):
        self.attempt = [(0, self.df.iloc[0].ct, self.roc)]

    def preprocess(self, df):
        delta = df.iloc[[0, -1]].diff()
        dsoc = delta.iloc[-1].soc
        dt = delta.iloc[-1].t
        if dsoc <= 0:
            raise ValueError(f'dsoc = {dsoc} for {self.id}')
        elif dt <= 0:
            raise ValueError(f'dt = {dt} for {self.id}')
        else:
            self.roc = (dt/60)/dsoc
            df['ct'] = (100-df.soc) * self.roc
        return df

    def obs(self, index=0):
        row = self.df.iloc[index]
        return [row[x[0]] for x in self.state]

    def step(self, action, start, stop):
        seg = self.df[start:stop]
        roc = action[0]

        obs = seg.iloc[-1]
        p = (100-obs.soc) * roc
        a = obs.ct
        d = abs(p-a)
        reward = 1 - min(1, d/10)

        self.attempt.append((obs.name, p, roc))
        obs = self.obs(obs.name)
        done = stop >= len(self.df)
        return obs, reward, done, {'a': a, 'p': p, 'pr': roc}
