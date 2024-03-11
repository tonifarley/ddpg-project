import torch
import os, boto3, json, tarfile
import numpy as np
from io import BytesIO
from erev.model import Actor

def initialize(device, fn):
    model = Actor(7, 1).to(device)
    model.load_state_dict(torch.load(fn))
    return model.float().to(device)

def preprocess(device, data):
    data = [float(x) for x in data.split(',')]
    tensor = torch.from_numpy(np.array(data))
    return tensor.to(device)

def inference(device, model, data):
    model = model.to(device)
    input_data = data.float().to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_data)
        return output

def postprocess(data):
    if type(data) == torch.Tensor:
        data = data.detach().cpu().numpy().tolist()
    return data

def test(vid, modelfn, states):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize(device, modelfn)
    results = []
    for obs in states:
        result = preprocess(device, obs)
        result = inference(device, model, result)
        result = postprocess(result)
        results.append(result)
    return results