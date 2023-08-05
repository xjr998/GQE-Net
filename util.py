import numpy as np
import torch
import torch.nn.functional as F
import random
import math


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def MSE(pred, gold):
    ''' Calculate MSE loss '''
    gold = gold.contiguous()
    loss_fn = torch.nn.MSELoss()

    loss = loss_fn(pred, gold)
    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    xyz = torch.tensor(xyz)    # from numpy to tensor
    xyz = xyz.to(torch.float)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    centroids = centroids.detach().numpy()      # from tensor to numpy
    return centroids


def search_knn(c, x, k):
    pairwise_distance = torch.sum(torch.pow((c - x), 2), dim = -1)
    idx = (-pairwise_distance).topk(k = k, dim = -1)[1]   # (batch_size, num_points, k)
    return idx


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mse2psnr(mse):
    psnr = 10*math.log10(255*255/mse)
    return psnr


def rgb2yuv(rgb):
    # PointNum=rgb.shape[0]
    yuv = np.zeros(rgb.shape)
    yuv[:, 0] = 0.2126*rgb[:, 0]+0.7152*rgb[:, 1]+0.0722*rgb[:, 2]
    yuv[:, 1] = -0.1146*rgb[:, 0]-0.3854*rgb[:, 1]+0.5000*rgb[:, 2] + 128
    yuv[:, 2] = 0.5000*rgb[:, 0]-0.4542*rgb[:, 1]-0.0458*rgb[:, 2] + 128
    # for i in range(PointNum):
    #     yuv[i, 0]=0.2126*rgb[i,0]+0.7152*rgb[i,1]+0.0722*rgb[i,2];
    #     yuv[i, 1]=-0.1146*rgb[i,0]-0.3854*rgb[i,1]+0.5000*rgb[i,2]+128;
    #     yuv[i, 2]=0.5000*rgb[i,0]-0.4542*rgb[i,1]-0.0458*rgb[i,2]+128;
    yuv = yuv.astype(np.float32)
    return yuv


def yuv2rgb(yuv):
    # PointNum=yuv.shape[0]
    yuv[:, 1] = yuv[:, 1] - 128
    yuv[:, 2] = yuv[:, 2] - 128
    rgb = np.zeros(yuv.shape)
    rgb[:, 0] = yuv[:, 0] + 1.57480 * yuv[:, 2]
    rgb[:, 1] = yuv[:, 0] - 0.18733 * yuv[:, 1] - 0.46813 * yuv[:, 2]
    rgb[:, 2] = yuv[:, 0] + 1.85563 * yuv[:, 1]
    return rgb

def cal_psnr(input1, input2):
    # img1 = input1.astype(np.float64)
    # img2 = input2.astype(np.float64)
    img1 = input1.to(torch.float64)
    img2 = input2.to(torch.float64)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    return psnr



def eval_new(opt, model, input):
    model.eval()
    preds = model(input)
    return preds


def log_string(log, out_str):
    log.write(out_str + '\n')
    log.flush()
    print(out_str)



def cal_mean(list):       # 对于重复使用的点计算加权平均值
    number = len(list)
    idx = [index for index in range(number) if list[index].size != 1]       #  找出重复使用的点的索引
    for i in idx:
        i_temp = list[i]
        list[i] = torch.mean(i_temp, dim=0)
    return list


if __name__ == "__main__":
    c = torch.randn(2,3)
    x = torch.randn(5,3)
    print(x, c)
    # idx = search_knn(c, x, 1)
    # print(idx.size())
    # print(x[idx])
    # print(torch.sum(x[idx]-c))
    # print(x[idx].size())
    
    print(np.clip(np.round(yuv2rgb(c)), 0, 255))

    
