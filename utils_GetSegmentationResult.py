from mit_semseg.config import cfg
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.dataset import TestDataset
from mit_semseg.lib.nn import user_scattered_collate
from mit_semseg.utils import colorEncode
from PIL import Image
import torch
from torch import nn
import os
import argparse
from argparse import Namespace
from torch.autograd import Variable
import numpy as np
import collections
from scipy.io import loadmat
import csv
from skimage import io
import matplotlib.pyplot as plt

colors = loadmat('./data/color150.mat')['colors']
names = {}
with open('./data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]
names_reversed = {value : key for (key, value) in names.items()}


def get_label_op(index):
    return names[index + 1]

def get_labeled_prediction(pred):
    return [[get_label_op(pred[i][j]) for j in range(pred.shape[1])] for i in range(pred.shape[0])]

def search_for_class(class_name, prediction):
    assert class_name in names_reversed, "{cl} is not a valid class".format(cl=class_name)
    index = names_reversed[class_name]
    return np.equal(prediction, index-1)

def search_for_classes(prediction, *args):
    result = np.equal(prediction, -1) # all False
    dim0 = len(result)
    dim1 = len(result[0])
    print(args)
    for class_name in args[0]:
        assert class_name in names_reversed, "{cl} is not a valid class".format(cl=class_name)
        index = names_reversed[class_name]
        mask = np.equal(prediction, index-1)
        result = [[result[i][j] or mask[i][j] for j in range(dim1)] for i in range(dim0)]
    return result

def visualize_result(data, pred, cfg):
    (img, info) = data

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(
        os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))

def as_numpy(obj):
    if isinstance(obj, collections.Sequence):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy()
    else:
        return np.array(obj)

def test(segmentation_module, loader, gpu):
    segmentation_module.eval()

    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
            #print("pred:", get_classes(pred))
            return pred

        # visualization
        # visualize_result(
        #   (batch_data['img_ori'], batch_data['info']),
        #    pred,
        #    cfg
        #)


def main(cfg, gpu):
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_test = TestDataset(
        cfg.list_test,
        cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    return test(segmentation_module, loader_test, gpu)

def predict(imgPath):
    args = Namespace(imgs=imgPath,
                     cfg="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
                     opts=None)
    cfg.merge_from_file(args.cfg)
    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

    # assert os.path.exists(cfg.MODEL.weights_encoder) and \
    #    os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    imgs = [args.imgs]
    assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    cfg.list_test = [{'fpath_img': x} for x in imgs]

    if not os.path.isdir(cfg.TEST.result):
        os.makedirs(cfg.TEST.result)

    prediction = main(cfg, None)

    return prediction

def get_class_mask(imgPath, className):
    assert className in names_reversed, "{cl} is not a valid class".format(cl=className)
    prediction = predict(imgPath)
    return search_for_class(className, prediction)

def get_mask_for_classes(imgPath, *args):
    prediction = predict(imgPath)
    return search_for_classes(prediction, *args)

if __name__ == '__main__':
    mask = get_mask_for_classes("./RGB.jpg", "tree", "grass")
    plt.imshow(mask)
    plt.show()



