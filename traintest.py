#!/opt/conda/bin/python3.6
import sys, os, boto3, math, random, torch, io, shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from datetime import datetime

import serve, model
from utils import get_data, put_data, email_admin
from flexenv import FlexEnv
from flexcontexts import LSET, RANGE, ESOC
from ddpg import Agent

def main(prefix, context, resume=False):
    stepsize = 100
    epochs = stepsize * 10
    trainsize = 20
    testsize = 0
    mintrip = 5000
    seed = 0
    idx = 0
    qa = False
    service = context.__name__.lower()

    print(f'Starting {service}, qa={qa}')

    cpath = f'tmp/models/{prefix}/{service}-checkpoint.pth'
    spath = cpath.replace('checkpoint', 'serve')

    trainers, testers, nsteps = setup(prefix, context)
    print(f'Setup {len(trainers)} trainers, {len(testers)} testers, {nsteps} steps')

    env = FlexEnv(context, trainers, stepsize)
    agent = Agent(env, nsteps*epochs, seed=seed)
    state = None
    if resume:
        if os.path.exists(cpath):
            print(f'Resume training loading {cpath}')
            state = agent.load(cpath)
            agent.train()
            print(f'agent state={state}')
            m = state['modelpath'].split('-n')[1].split('e')[1].split('ss')[0]
            m = int(m) + epochs
        else:
            print(f'Resume training cannot find {cpath}, using new agent.')
            m = epochs
        config = f'{service}-n{len(trainers)}e{m}-'

    print(f'Starting {prefix}: {config}')
    modelpath = f'tmp/models/{prefix}'
    os.makedirs(modelpath, exist_ok=True)

    rewards, records = train(env, agent, epochs)

    if testers:
        testrecords, score = test(env, agent)
        if score > maxscore:
            maxscore = score
            bestpath = modelpath
            bestrewards = rewards
            bestrecords = records
            besttestrecords = testrecords
    else:
        bestrewards = rewards
        bestrecords = records

    modelpath = f'{modelpath}/{config}r{bestreward:.2f}.pth'
    with open(modelpath, 'wb') as f:
        f.write(bestmodel.getbuffer())
        print(f'Saved best model to {modelpath}')

    if bestpath:
        if not qa:
            agent = Agent(env)
            state = agent.load(bestpath)
            save = True
            if os.path.exists(cpath):
                if maxscore < currentscore:
                    print('Not saving model, new score {} < {}'.format(
                        maxscore, currentscore))
                    save = False
            if save:
                if os.path.exists(cpath):
                    shutil.copyfile(cpath, cpath.replace('pth', 'old'))
                    print(f'Copied {cpath} to old')
                if os.path.exists(spath):
                    shutil.copyfile(spath, spath.replace('pth', 'old'))
                    print(f'Copied {spath} to old')

                agent.save(bestpath, cpath, state['reward'], maxscore)
                print(f'Saved {bestpath} to {cpath}')
                torch.save(agent.actor.state_dict(), spath, 
                    _use_new_zipfile_serialization=False
                )
                shutil.copyfile(bestpath, spath)
                print(f'Saved state dict {bestpath} to {spath}')

                with open(cpath.rsplit('-', 1)[0]+'latest.txt', 'w') as f:
                    f.write(f'{bestpath}-s{maxscore:.2f}')
                try:
                    put_data(session, 'wave', f'images/{prefix}', f'tmp/images/{prefix}')
                    put_data(session, 'wave', f'models/{prefix}', f'tmp/models/{prefix}')
                except:
                    print('Error: Did not copy to S3')

        print(f'Model path = {bestpath}')
        plot(prefix, bestpath, bestrewards, bestrecords)
        plot(prefix, f'{bestpath}-s{maxscore:.2f}', [], besttestrecords, True)

    else:
        plot(prefix, 'test/lom', bestrewards, bestrecords)

def data_paths(prefix):
    dd = f'tmp/data'
    get_data(session, 'exdata', prefix, dd)
    paths = sorted(
        [f'{dd}/{prefix}/{x}' for x in os.listdir(f'{dd}/{prefix}') if not x.startswith('.')],
        reverse=True
    )
    return paths

def make_contexts(paths, context):
    result = []
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            cname = '/'.join(path.split('/')[-2:]).split('.')[0]
            result.append(context(cname, df))
    return result

def setup(prefix, context, random=False):
    paths = data_paths(prefix)
    print(f'Starting setup, {len(paths)} paths')

    trainers = []
    rocs = {}
    nsteps = 0
    size = trainsize+testsize
    idx = 0
    if random:
        random.shuffle(paths)

    while (len(trainers) < size) and (idx < len(paths)):
        path = paths[idx]
        idx += 1
        df = pd.read_csv(path)
        require = [x[0] for x in context.state if x[0] not in ['fuel', 'lset']]
        if df[require].isnull().values.any():
            print(f'train.setup skipping nans {path}')
        else:
            cname = '/'.join(path.split('/')[-2:]).split('.')[0]
            dsoc = df.soc.max()-df.soc.min()
            if len(df) > mintrip and df.soc[0] > 50 and dsoc > 20:
                ctx = context(cname, df)
                if service == 'range':
                    if (3 > ctx.roc > .5):
                        rocs[ctx.id] = ctx.roc
                        nsteps += math.ceil(len(df)/stepsize)
                        trainers.append(ctx)
                    else:
                        print(f'train.setup skipping roc {ctx.roc} {ctx.id}')
                else:
                    nsteps += math.ceil(len(df)/stepsize)
                    trainers.append(ctx)

    if type(context) == RANGE:
        df = pd.DataFrame.from_dict(rocs, orient='index')
        df.to_csv('rocs.csv')
        print('train.setup data collected rocs describe')
        print(df.describe())

    return trainers[testsize:], trainers[:testsize], nsteps

def train(env, agent, epochs):
    rewards = []
    records = {}
    if qa:
        steps = {'to_env':[], 'before_noise': [], 'noise':[], 'env/ctx': [], 'rewards':[]}
    for i in range(epochs):
        obs = env.reset(idx)
        erewards = []
        done = False
        while not done:
            action, a, n = agent.act(obs)
            nextobs, reward, done, info = env.step(action)
            agent.update(obs, action, reward, nextobs, int(done))

            if qa:
                steps['to_env'].append(action[0])
                steps['before_noise'].append(a[0])
                steps['noise'].append(n[0])
                steps['env/ctx'].append(info['aa'][0])
                steps['rewards'].append(reward)

            obs = nextobs
            erewards.append(reward)

        ereward = np.mean(erewards)
        rewards.append(ereward)

    for ctx in env.ctxs:
        attempt = pd.DataFrame(ctx.attempt, columns=('index', 'attempt')).set_index('index')
        records[ctx.id] = attempt.join(ctx.goal).round(3)

    print(f'Training completed {i+1} runs, best/mean reward {bestreward:.3f}/{np.mean(rewards):.3f}')

    if qa:
        df = pd.DataFrame(steps)
        df.to_csv('traincheck.csv', index_label='step')
        df[['to_env', 'env/ctx', 'noise', 'rewards']].round(3).plot(subplots=True)
        print('--> train show steps')
        plt.show()

    return rewards, records

def test(prefix, testers, context, modelpath, actor=None):
def test(env, agent):
    print(f'Testing {service} {prefix} on {len(testers)} contexts')
    print(f'Actor is {actor}')
    env = FlexEnv(context, testers, stepsize)
    print('Set env action space', env.action_space.high, env.action_space.low)
    event = {'service': service, 'modelpath': modelpath, 'actor': actor, 'state': env.reset(idx)}

    ids = [x.id.split('/')[1] for x in testers]
    print(f'Testing {ids}')
    if service == 'range':
        rocs = [x.roc for x in testers]
        mroc = np.mean(rocs)
        print(f'rocs={rocs}, mean={mroc}')

    result = {'predicted':[], 'p':[], 'actual':[], 'soc':[]}

    done = False
    scores = []
    while not done:
        soc = event['state'][-1]
        action = np.array([serve.handler(event, {})])
        nextobs, score, done, info = env.step(action, convert=False)
        result['actual'].append(info['a'])
        if service == 'range':
            result['predicted'].append(soc * action[0])
        else:
            result['predicted'].append(action[0])
        result['p'].append(info['p']) # This is plotted in the end
        result['soc'].append(soc)
        scores.append(score)
        event['state'] = nextobs

    meanscore = np.mean(scores)
    print(f'Test best/avg scores {np.max(scores):.3f}/{meanscore:.3f}')

    df = pd.DataFrame(result)
    df.to_csv('testcheck.csv', index_label='step')

    if qa:
        df[['actual', 'predicted', 'p']].round(3).plot()
        print('--> test show steps')
        plt.show()

    records = {}
    for ctx in env.ctxs:
        attempt = pd.DataFrame(ctx.attempt, columns=('index', 'attempt')).set_index('index')
        records[ctx.id] = attempt.join(ctx.goal).round(3)

        # TMP output for review
        df = ctx.df.join(attempt).rename(columns={'attempt': 'predicted'})
        os.makedirs('test/150488', exist_ok=True)
        df.to_csv(f'test/{ctx.id}.csv')

    return records, meanscore

def plot(prefix, modelpath, rewards, records, test=False):
    perpage = 20
    pages = math.ceil(len(records)/perpage)
    fn = modelpath.rsplit('/', 1)[1]
    if test:
        title = f'Vehicle {prefix} Testing\n{fn}'
    else:
        title = f'Vehicle {prefix} Training\n{fn}'
    records = list(records.items())

    dd = f'tmp/images/{prefix}'
    os.makedirs(dd, exist_ok=True)

    for page in range(pages):
        s = page*perpage
        data = records[s:s+perpage]
        ntrips = len(data)

        if ntrips > 3:
            dimx = math.ceil(math.sqrt(ntrips))
            dimy = math.ceil(ntrips/dimx)
        else:
            dimx = 1
            dimy = ntrips

        fig, axs = plt.subplots(dimx, dimy, sharex=True, sharey=True)
        fig.suptitle(title)

        if ntrips > 1:
            axs = axs.flat
        else:
            axs = [axs]

        for i, ax in enumerate(axs):
            if i < len(data):
                k, df = data[i]
                ax.plot(df['attempt'], label='model')
                m = {'rr', 'lset', 'esoc'}.intersection(df.columns)
                m = m.pop() if m else 'soc'
                ax.plot(df[m], label='actual')
                ax.set(xlabel='time', ylabel=m)
                ax.set_title(k.split('/')[1], fontsize=10)
                ax.label_outer()
                handles, labels = ax.get_legend_handles_labels()
            else:
                ax.remove()

        fig.legend(handles, labels, loc='upper right', frameon=False)
        plt.tight_layout(pad=.5)
        if not qa:
            if test:
                fig.savefig(f'{dd}/{fn}-test-{page}.png')
            else:
                fig.savefig(f'{dd}/{fn}-{page}.png')
    if qa:
        print(f'--> plot show results test={test}')
        plt.show()

    if not test:
        fig, ax = plt.subplots()
        ax.plot(rewards)
        ax.set(title=title, xlabel='epoch', ylabel='reward')
        if qa:
            print(f'--> plot show rewards test={test}')
            plt.show()
        else:
            fig.savefig(f'{dd}/{fn}-rewards.png')

def test_save(bestpath, spath):
    state = '.09,29.67,-95.33,-5.09,99' # 163.34
    data = [float(x) for x in state.split(',')]
    for mp in (bestpath, spath):
        print(mp)
        actor = model.Actor(5, 1)
        actor.load_state_dict(torch.load(mp))
        actor.eval()
        with torch.no_grad():
            output = actor(torch.tensor(data).float())
            result = output.detach().cpu().numpy()
        result = model.convert(result[0], service='range')
        if 'range' in mp:
            print(99*result)
        else:
            print(result)
