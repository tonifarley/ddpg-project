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
from flexcontexts import LSET, RANGE, ESOC, CHARGE
from ddpg import Agent
stepsize = 10
mintrip = 200

epochs = 5
maxruns = 30
trainsize = 5
testsize = 1
startfile = 0
seed = 0
idx = 0
service = ''
qa = False

def train(env, agent, prefix, trainers, context, nsteps, state=None):
    print(f'Training {prefix} on {len(trainers)} contexts')

    if state:
        m = state['modelpath'].split('-n')[1].split('e')[1].split('ss')[0]
        m = int(m) + epochs
        config = f'{service}-n{len(trainers)}e{m}ss{stepsize}-'
    else:
        config = f'{service}-n{len(trainers)}e{epochs}ss{stepsize}-'

    ts = datetime.now()
    print(f'Starting {prefix}: {config}: {ts}')
    modelpath = f'tmp/models/{prefix}'
    os.makedirs(modelpath, exist_ok=True)

    rewards = []
    records = {}
    bestreward = 0
    bestmodel = io.BytesIO()
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

        if ereward > bestreward:
            bestreward = ereward
            torch.save(agent.actor.state_dict(), bestmodel, 
                _use_new_zipfile_serialization=False
            )

    modelpath = f'{modelpath}/{ts:%y%m%d.%I.%M}-{config}r{bestreward:.2f}.pth'

    for ctx in env.ctxs:
        if service.startswith('range') or service == 'charge':
            attempt = pd.DataFrame(
                ctx.attempt, columns=('index', 'predicted', 'predicted_roc')
            ).set_index('index')
        else:
            attempt = pd.DataFrame(
                ctx.attempt, columns=('index', 'predicted')
            ).set_index('index')
        records[ctx.id] = attempt.join(ctx.goal).round(3)

    print(f'Training completed {i+1} runs, best/mean reward {bestreward:.3f}/{np.mean(rewards):.3f}')

    with open(modelpath, 'wb') as f:
        f.write(bestmodel.getbuffer())
        print(f'Saved best model to {modelpath}')

    print(f'Completion time {datetime.now() - ts}')

    if qa:
        df = pd.DataFrame(steps)
        df.to_csv('traincheck.csv', index_label='step')
        df[['to_env', 'env/ctx', 'noise', 'rewards']].round(3).plot(subplots=True)
        print('--> train show steps')
        plt.show()

    return modelpath, rewards, records

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
                ax.plot(df['predicted'], label='model')
                m = {'rr', 'lset', 'esoc', 'ct'}.intersection(df.columns)
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

def test(prefix, testers, context, modelpath, actor=None):
    print(f'Testing {service} {prefix} on {len(testers)} contexts')
    print(f'Actor is {actor}')
    env = FlexEnv(context, testers, stepsize)
    print('Set env action space', env.action_space.high, env.action_space.low)
    event = {'service': service, 'modelpath': modelpath, 'actor': actor, 'state': env.reset(idx)}

    ids = [x.id.split('/')[1] for x in testers]
    print(f'Testing {ids}')
    if service.startswith('range'):
        rocs = [x.roc for x in testers]
        print(f'ROCs range testing = ({np.min(rocs)}, {np.max(rocs)})')

    result = {'predicted':[], 'p':[], 'actual':[], 'soc':[]}

    done = False
    scores = []
    while not done:
        soc = event['state'][-1]
        action = np.array([serve.handler(event, {})])
        nextobs, score, done, info = env.step(action, convert=False)
        result['actual'].append(info['a'])
        if service.startswith('range'):
            result['predicted'].append(soc * action[0])
        elif service == 'charge':
            result['predicted'].append((100-soc) * action[0])
        else:
            result['predicted'].append(action[0])
        result['p'].append(info['p'])
        result['soc'].append(soc)
        scores.append(score)
        event['state'] = nextobs

    meanscore = np.mean(scores)
    print(f'Test best/avg scores {np.max(scores):.3f}/{meanscore:.3f}')

    if qa:
        df = pd.DataFrame(result)
        df.to_csv('testcheck.csv', index_label='step')
        df[['actual', 'predicted', 'p']].round(3).plot()
        print('--> test show steps')
        plt.show()

    records = {}
    for ctx in env.ctxs:
        if service.startswith('range') or service == 'charge':
            attempt = pd.DataFrame(
                ctx.attempt, columns=('index', 'predicted', 'predicted_roc')
            ).set_index('index')
        else:
            attempt = pd.DataFrame(
                ctx.attempt, columns=('index', 'predicted')
            ).set_index('index')
        dfc = attempt.join(ctx.goal).round(3)
        dfc = dfc.join(ctx.df[['t', 'soc']], how='inner')
        records[ctx.id] = dfc

    return records, meanscore

def setup(prefix, paths, context, startfile=0):
    trainers = []
    rocs = []
    nsteps = 0
    size = trainsize+testsize
    idx = 0
    print(f'Setting up service {service} with {len(paths)} paths')
    random.shuffle(paths)
    paths = paths[startfile:]
    while (len(trainers) < size) and (idx < len(paths)):
        path = paths[idx]
        idx += 1
        df = pd.DataFrame({})
        try:
            df = pd.read_csv(path)
        except:
            print(f'train.setup skipping read fail {path}')

        if not df.empty:
            require = [x[0] for x in context.state if x[0] not in ['fuel', 'lset']]
            if df[require].isnull().values.any():
                print(f'train.setup skipping nans {path}')
            else:
                cname = '/'.join(path.split('/')[-2:]).split('.')[0]
                dsoc = df.soc.max()-df.soc.min()
                if len(df) > mintrip:
                    if (service =='charge' and df.soc.values[-1] == 100) or (
                        df.soc[0] > 50 and dsoc > 10):

                        ctx = context(cname, df)
                        if service.startswith('range'):
                            if (3 > ctx.roc > .5):
                                rocs.append(ctx.roc)
                                nsteps += math.ceil(len(df)/stepsize)
                                trainers.append(ctx)
                            else:
                                print(f'train.setup skipping roc {ctx.roc} {ctx.id}')
                        else:
                            nsteps += math.ceil(len(df)/stepsize)
                            trainers.append(ctx)
                    else:
                        print(f'train.setup skipping other {path}')
                else:
                    print(f'train.setup skipping short {path}')
        else:
            print(f'Empty df at path {path}')

    if service.startswith('range'):
        print(f'ROCs range training = ({np.min(rocs)}, {np.max(rocs)})')

    return trainers[testsize:], trainers[:testsize], nsteps

def trials(prefix, context):
    global trainsize, epochs, startfile
    score = 92
    results = []
    for i in range(1,5):
        epochs = i
        s = main(prefix, context, currentscore=score)
        score = max(s, score)
        results.append(f'{i}\t{s}')
    print('\n'.join(results))

def main(prefix, context, resume=False, currentscore=0):
    if 'AWS_BATCH_JOB_ID' in os.environ:
        session = boto3.Session(region_name='us-east-2')
    else:
        session = boto3.Session(profile_name='prodef')

    global service
    service = context.__name__.lower()
    print(f'Starting {service}')

    dd = f'tmp/data'
    get_data('exdata', prefix, dd)
    paths = sorted(
        [f'{dd}/{prefix}/{x}' for x in os.listdir(f'{dd}/{prefix}') if not x.startswith('.')],
        reverse=True
    )
    print(f'Found {len(paths)} data files')

    maxscore = 0
    if paths:
        runcount = 0
        bestpath = ''
        bestrewards = []
        bestrecords = {}
        besttestrecords = {}
        cpath = f'tmp/models/{prefix}/{service}-checkpoint.pth'
        spath = cpath.replace('checkpoint', 'serve')

        while maxscore <= currentscore and runcount < maxruns:
            runcount += 1
            print(f'============== Starting {runcount} =============')

            trainers, testers, nsteps = setup(prefix, paths, context)
            print(f'Setup {len(trainers)} trainers, {len(testers)} testers, {nsteps} steps')

            if len(trainers) >= trainsize and len(testers) >= testsize:
                env = FlexEnv(context, trainers, stepsize)
                agent = Agent(env, nsteps*epochs, seed=seed)
                state = None
                modelpath, rewards, records = train(env, agent, prefix, trainers, context, nsteps, state)
                if testers:
                    testrecords, score = test(prefix, testers, context, modelpath)
                    if score > maxscore:
                        maxscore = score
                        bestpath = modelpath
                        bestrewards = rewards
                        bestrecords = records
                        besttestrecords = testrecords
                else:
                    bestrewards = rewards
                    bestrecords = records
            else:
                sys.exit()

        print(f'Finished after {runcount} runs with best test avg {maxscore}')

        if bestpath:
            if not qa:
                agent = Agent(env)
                state = agent.load(bestpath)
                if os.path.exists(spath):
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

            print(f'Model path = {bestpath}')
            plot(prefix, bestpath, bestrewards, bestrecords)
            plot(prefix, f'{bestpath}-s{maxscore:.2f}', [], besttestrecords, True)

        else:
            plot(prefix, 'test/lom', bestrewards, bestrecords)

    else:
        print(f'No data in {dd}')

    return maxscore

def test_save(bestpath, spath):
    state = '.09,29.67,-95.33,-5.09,99' # 163.34
    data = [float(x) for x in state.split(',')]
    t = []
    for mp in (bestpath, spath):
        actor = model.Actor(5, 1)
        actor.load_state_dict(torch.load(mp))
        actor.eval()
        with torch.no_grad():
            output = actor(torch.tensor(data).float())
            result = output.detach().cpu().numpy()
        result = model.convert(result[0], service='range')
        t.append(result)
    assert t[0] == t[1], f'spath in error, {t[0]} != {t[1]}'

def make_contexts(paths, context):
    result = []
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            cname = '/'.join(path.split('/')[-2:]).split('.')[0]
            result.append(context(cname, df))
    return result
