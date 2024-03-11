import sys
import numpy as np
from itertools import cycle

class FlexEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    """
    :param context: a class with defined action and state space
    :param ctxs: a list of context objects
    :stepsize: number of rows to process with each action
    """
    def __init__(self, context, ctxs, stepsize):
        super(FlexEnv, self).__init__()
        self.context = context
        self.ctxs = ctxs
        self.stepsize = stepsize

        actions = context.actions
        if type(actions) == list:
            self.action_space = gym.spaces.Discrete(len(actions))
        elif type(actions) == range:
            self.action_space = gym.spaces.Box(float(min(actions)), float(max(actions)), (1,))

        space = np.array(list(map(list, zip(*context.state))))[1:].astype(np.float32)
        self.observation_space = gym.spaces.Box(space[0], space[1])

    def reset(self, i=None):
        if i:
            self.i = i
        else:
            self.i = np.random.randint(self.stepsize)
        self.ictxs = iter(self.ctxs)
        self.ctx = next(self.ictxs)
        for d in self.ctxs:
            d.reset()
        return np.array(self.ctx.obs())

    def convert(self, action):
        x = (self.action_space.high - self.action_space.low) / 2.
        y = (self.action_space.high + self.action_space.low) / 2.
        return x * action + y

    # Set convert to False for testing serve (which already converted)
    def step(self, action, convert=True):
        if convert:
            action = self.convert(action)
        nexti = self.i + self.stepsize
        obs, reward, done, info = self.ctx.step(action, self.i, nexti)
        if done:
            self.i = np.random.randint(self.stepsize)
            self.ctx = next(self.ictxs, None)
        else:
            self.i = nexti
        info['aa'] = action
        return np.array(obs), reward, self.ctx==None, info

    def render(self, mode='human', close=False):
        if 'render' in dir(self.context):
            self.context.render()
        else:
            print(vars(self.context))

    def seed(self):
        if 'seed' in dir(self.context):
            self.context.seed()

    def close(self):
        if 'close' in dir(self.context):
            self.context.close()

    def configure(self):
        if 'configure' in dir(self.context):
            self.context.configure()

    def convert(action, ctx=None, service=None):
        if ctx:
            x = (self.action_space.high - self.action_space.low) / 2.
            y = (self.action_space.high + self.action_space.low) / 2.
        else:
            high = actions[service][1]
            low = actions[service][0]
            x = (high - low) / 2.
            y = (high + low) / 2.
        return x * action + y
