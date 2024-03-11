try:
    import unzip_requirements
except ImportError:
    pass
import os, boto3, torch
import model, oldmodel

# TODO only print if prod or testing, not training/testings
def handler(event, context):
    # print(f'Serve called with {event}, {context}')
    if context or ('AWS_BATCH_JOB_ID' in os.environ):
        session = boto3.Session(region_name='us-east-2')
    else:
        session = boto3.Session(profile_name='prodef')

    result = ''
    if event.get('actor'):
        actor = event['actor']
    else:
        if event['service'] == 'lset':
            actor = oldmodel.Actor(7, 1)
        elif event['service'] == 'charge':
            actor = model.Actor(6, 1)
        else:
            actor = model.Actor(5, 1)
        mp = event.get('modelpath', None)
        # print(f'Serve using model {mp}')
        if mp:
            if os.path.exists(mp):
                # print(f'Serve using cached model {mp}')
                actor.load_state_dict(torch.load(mp))
            else:
                # print('Serve downloading from S3')
                try:
                    session.resource('s3').meta.client.download_file('wave', mp, '/tmp/model.pth')
                    actor.load_state_dict(torch.load('/tmp/model.pth'))
                except:
                    print(f'Cannot download mp {mp}')
                    result = 'Error getting model'
        else:
            result = 'Error, no actor or modelpath in context'

    if not result:
        if event.get('seed'):
            torch.manual_seed(event['seed'])
        actor.eval()
        data = event['state']
        if type(data) == str:
            data = [float(x) for x in data.split(',')]
        with torch.no_grad():
            output = actor(torch.tensor(data).float())
            result = output.detach().cpu().numpy()
        result = model.convert(result[0], service=event['service'])

    # print(f'Serve returning {result}')
    return result
