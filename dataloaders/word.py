'''
    Author: Sungguk Cha
    eMail : navinad@naver.com
'''

from mypath import Path

import fasttext as ft
import os


def get_maps(dataset):
    classes = {
        'pascal':['aeroplane','bicycle','bird','boat',
                 'bottle','bus','car','cat',
                 'chair','cow','diningtable','dog',
                 'horse','motorbike','person','pottedplant',
                 'sheep','sofa','train','tvmonitor']
        }
    maps = {0:None}
    for i in range(len(classes[dataset])):
        maps[i+1] = classes[dataset][i]
    return maps

def build_classifier(dataset, norm):
    model_path = os.path.join(Path.db_root_dir('wiki'), 'wiki.en.bin') #이거 경량화해서 저장해둬야해
    model = ft.load_model(model_path)

    maps = get_maps(dataset)
    embeddings = []
    for i in range(1, len(maps)):
        embeddings.append(model[maps[i]])
    
    return embeddings
