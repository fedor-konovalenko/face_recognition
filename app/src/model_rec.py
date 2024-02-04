import torch
import torchvision.transforms.v2 as transforms
import logging
from PIL import Image
import warnings
from torchvision.models import densenet121
from torch import nn
from cropper import MyModel, Identity
import numpy as np
from itertools import combinations

warnings.filterwarnings("ignore")

m_logger = logging.getLogger(__name__)
m_logger.setLevel(logging.DEBUG)
handler_m = logging.StreamHandler()
formatter_m = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
handler_m.setFormatter(formatter_m)
m_logger.addHandler(handler_m)

IMG_SIZE = 224
F_SIZE = 128
DEVICE = "cpu"
CROPPER = MyModel()


def embedder(params_path: str, device: str):
    """load model and weights"""
    emb_model = densenet121(weights=None)
    in_features = 1024
    out_features = 500
    emb_model.classifier = nn.Linear(in_features, out_features)
    emb_model = emb_model.to(device)
    emb_model.load_state_dict(torch.load(params_path, map_location=torch.device(device)))
    emb_model.classifier = Identity()
    return emb_model


def crop(img, preds):
    coord_x = preds[0][:2].cpu().numpy()
    coord_y = preds[0][2:].cpu().numpy()
    dist = abs(coord_x[0] - coord_x[1])
    delta = abs(coord_y[0] - coord_y[1])
    angle = np.arctan(delta / dist) * 180 / np.pi
    img = transforms.functional.rotate(img, -angle)
    k = int(60 * (64 / dist))
    new_size = (k * img.size()[1], 120)
    crop_1 = transforms.CenterCrop(new_size)
    img = crop_1(img)
    crop_2 = transforms.CenterCrop((128, 128))
    img = crop_2(img)
    return img


def predict(f_path: list):
    """detecting function TODO: Fix the similarity formula"""
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    trained_model = CROPPER.to(DEVICE)
    trained_model.load_state_dict(torch.load('enet_reg.pth', map_location=torch.device(DEVICE)))
    transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True)
                                      ])
    emb_maker = embedder('densenet.pth', DEVICE)
    embs = []
    for fp in f_path:
        image = Image.open(fp)
        image = transformer(image)
        with torch.no_grad():
            preds = trained_model(image.unsqueeze(0).to(DEVICE))
        result = crop(image, preds)
        with torch.no_grad():
            embs.append(emb_maker(result.to(DEVICE).unsqueeze(0)))
    sim = []
    if len(embs) == 0:
        status = 'Fail'
        m_logger.warning(f'some problems with recognition')
        sim = []
    else:
        status = 'OK'
        m_logger.info(f'recognition success')
        for e1, e2 in combinations(embs, 2):
            e1 = nn.functional.normalize(e1)
            e2 = nn.functional.normalize(e2)
            sim.append(round((1 - float(cos(e1, e2))) * 1000, 2))
    return status, sim
