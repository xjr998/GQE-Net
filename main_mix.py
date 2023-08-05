from __future__ import print_function
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import *
from model_GQE_Net_final import GAPCN
from tqdm import tqdm
import datetime
import time
from torch.utils.data import DataLoader
from util import *
from sewar.full_ref import psnr

devices = "cuda:0"


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('copy main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('copy model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('copy util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('copy data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, io):
    daytime = datetime.datetime.now().strftime('%Y-%m-%d')  # year,month,day
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    channel = args.train_channel
    yuv_list = ['y', 'u', 'v']
    DATA_DIR = os.path.join(BASE_DIR, args.train_h5_txt)
    DATA_DIR_TEST = os.path.join(BASE_DIR, args.valid_h5_txt)
    logs_path = args.log_path + '/GQE-Net/' + daytime + '/' + yuv_list[channel]
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    model_path = args.pth_path + '/GQE-Net/' + daytime + '/' + yuv_list[channel]
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    txt_loss_mse = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))+'_loss.txt'
    txt_mse_path = os.path.join(logs_path, txt_loss_mse)
    txt_loss_psnr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))+'_loss_rgb.txt'
    txt_psnr_path = os.path.join(logs_path, txt_loss_psnr)
    txtValid_name_loss = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))+'_lossValid.txt'
    txt_lossValid_path = os.path.join(logs_path, txtValid_name_loss)
    txt_valid_loss_psnr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_valid_loss_rgb.txt'
    txt_valid_psnr_path = os.path.join(logs_path, txt_valid_loss_psnr)

    traindata, label = load_h5(DATA_DIR)
    dataset = torch.utils.data.TensorDataset(traindata, label)
    train_loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True)

    testdata, label = load_h5(DATA_DIR_TEST)
    # print(traindata)
    dataset_test = torch.utils.data.TensorDataset(testdata, label)
    test_loader = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=args.test_batch_size,
    shuffle=True,
    drop_last=True
)
    device = torch.device(devices if args.cuda else "cpu")
    model = GAPCN().to(device)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    if args.has_model:
        checkpoint = torch.load(args.model1_path)
        if channel == 1:
            checkpoint = torch.load(args.model2_path)
        elif channel == 2:
            checkpoint = torch.load(args.model3_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    criterion = MSE
    max_psnr = 0
    for epoch in range(args.epochs):
        epoch_loss = AverageMeter()
        epoch_loss_ori = AverageMeter()
        ####################
        # Train
        ####################
        args.lr = args.lr * (0.25 ** (epoch // 60))
        for p in opt.param_groups:
            p['lr'] = args.lr
        train_loss = 0.0
        count = 0.0
        model.train()
        with tqdm(total=(traindata.shape[0] - traindata.shape[0] % args.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch, args.epochs))
            for data, label in train_loader:
                data, label = data.to(device), label.to(device).squeeze()
                batch_size = data.size()[0]
                # qp = torch.zeros(batch_size)
                # for ii in range(batch_size):
                #     qp[ii] = data[ii, 0, 6]
                # data = data[:, :, :4]
                data = torch.cat((data[:, :, :3], torch.unsqueeze(data[:, :, channel + 3], dim=-1)), dim=-1)
                label = label[:, :, channel]

                if len(label.size()) == 2:
                    label = torch.unsqueeze(label, dim=-1)
                data = data.permute(0, 2, 1)

                opt.zero_grad()
                rec = data.permute(0, 2, 1)[:, :, 3:]
                data = torch.autograd.Variable(data, requires_grad=True)
                logits = model(data)

                loss = criterion(logits, label)
                loss_ori = criterion(rec, label)
                loss.backward()
                opt.step()
                epoch_loss.update(loss.item(), len(rec))
                epoch_loss_ori.update(loss_ori.item(), len(rec))

                count += batch_size
                train_loss += loss.item() * batch_size
                
                _tqdm.set_postfix(loss='{:.7f}'.format(epoch_loss.avg))
                _tqdm.update(len(data))
            
        scheduler.step()
        
        epoch_psnr = mse2psnr(epoch_loss.avg)
        epoch_psnr_ori = mse2psnr(epoch_loss_ori.avg)
        file_log = open(txt_mse_path, 'a')
        print('epoch:{} loss:{} loss_ori:{}'.format(epoch, epoch_loss.avg, epoch_loss_ori.avg), file=file_log)
        file_log.close()
        print('epoch:{}'.format(epoch), 'average loss:{}'.format(epoch_loss.avg))
        file_loss = open(txt_psnr_path, 'a')
        print('epoch:{}'.format(epoch), 'psnr:{}'.format(epoch_psnr), 'psnr_origin:{}'.format(epoch_psnr_ori), file=file_loss)

        file_loss.close()

        ####################
        # Test / Validating...
        ####################
        print('epoch:%d   validating...' % epoch)
        valid_epoch_loss = AverageMeter()
        valid_epoch_loss_ori = AverageMeter()

        model.eval()
        with tqdm(total=(testdata.shape[0] - testdata.shape[0] % args.test_batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch, args.epochs))
            for data, label in test_loader:
                data, label = data.to(device), label.to(device).squeeze()
                batch_size = data.size()[0]
                # qp = torch.zeros(batch_size)
                # for ii in range(batch_size):
                #     qp[ii] = data[ii, 0, 6]
                data = torch.cat((data[:, :, :3], torch.unsqueeze(data[:, :, channel + 3], dim=-1)), dim = -1)
                label = label[:, :, channel]
                if len(label.size()) == 2:
                    label = torch.unsqueeze(label, dim=-1)

                data = data.permute(0, 2, 1)
                with torch.no_grad():
                    logits = model(data)
                rec = data.permute(0, 2, 1)[:, :, 3:]
                # print(logits.size(), rec.size(), label.size())
                loss= criterion(logits, label)
                loss_ori= criterion(rec, label)

                valid_epoch_loss.update(loss.item(), len(data))
                valid_epoch_loss_ori.update(loss_ori.item(), len(data))

                _tqdm.set_postfix(loss='{:.7f}'.format(valid_epoch_loss.avg))
                _tqdm.update(len(data))
        valid_epoch_psnr = mse2psnr(valid_epoch_loss.avg)
        valid_epoch_psnr_ori = mse2psnr(valid_epoch_loss_ori.avg)
        # io.cprint(outstr)
        print('valid loss_ori:{}'.format(valid_epoch_loss_ori.avg), 'valid loss_preds:{}'.format(valid_epoch_loss.avg))
        fileValid_loss = open(txt_lossValid_path, 'a')
        print('epoch:{}'.format(epoch), 'valid average loss:{}'.format(valid_epoch_loss.avg))
        print('epoch:{}'.format(epoch), 'valid average loss:{}'.format(valid_epoch_loss.avg), 'valid average loss_ori:{}'.\
              format(valid_epoch_loss_ori.avg), file=fileValid_loss)

        fileValid_loss.close()
        file_valid_psnr = open(txt_valid_psnr_path, 'a')
        print('epoch:{}'.format(epoch), 'valid_psnr:{}'.format(valid_epoch_psnr), 'valid_psnr_origin:{}'.format(valid_epoch_psnr_ori),
              file=file_valid_psnr)
        if epoch >= 5 and valid_epoch_psnr-valid_epoch_psnr_ori > max_psnr:  # save the model with max PSNR promotion
            max_psnr = valid_epoch_psnr - valid_epoch_psnr_ori
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss,
            },
                '%s/model_%d.pth' % (model_path, epoch)
            )

        file_valid_psnr.close()



def test(args, io):
    device = torch.device(devices)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR_TEST = os.path.join(BASE_DIR, args.test_ply_txt)
    ori_path = os.path.join(BASE_DIR, args.test_ori_ply)
    rec_path = os.path.join(BASE_DIR, args.test_rec_ply)

    model1 = GAPCN().to(device)
    # checkpoint = torch.load(args.model1_path, map_location={'cuda:1': 'cuda:0'})
    checkpoint = torch.load(args.model1_path, map_location=torch.device(devices))
    model1.load_state_dict(checkpoint['model_state_dict'])
    model1 = model1.eval()

    model2 = GAPCN().to(device)
    # checkpoint = torch.load(args.model2_path, map_location={'cuda:1': 'cuda:0'})
    checkpoint = torch.load(args.model2_path, map_location=torch.device(devices))
    model2.load_state_dict(checkpoint['model_state_dict'])
    model2 = model2.eval()

    model3 = GAPCN().to(device)
    # checkpoint = torch.load(args.model3_path, map_location={'cuda:1': 'cuda:0'})
    checkpoint = torch.load(args.model3_path, map_location=torch.device(devices))
    model3.load_state_dict(checkpoint['model_state_dict'])
    model3 = model3.eval()

    textfile_name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_test_ply_trained_GAPCN.txt'
    LOG_FOUT = open(os.path.join(args.log_path_test, textfile_name), 'w')
    LOG_FOUT.write(str(args) + '\n')

    p_num = 2048
    iter = 0
    if not os.path.exists(args.pred_path):
        os.makedirs(args.pred_path)
    with open(DATA_DIR_TEST, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            log_string(LOG_FOUT, line)
            iter = iter + 1
            log_string(LOG_FOUT, 'sequence: %s, iter: %d' % (line, iter))
            if os.path.splitext(line)[1] == ".ply":
                ori_name = line.split('_r0')[0] + ".ply"
                pointcloud_ori = read_ply(os.path.join(ori_path, ori_name))  # [numpoint, 6]
                ori_color = rgb2yuv(pointcloud_ori[:, 3:]).astype(np.float32)
                pointcloud_rec = read_ply(os.path.join(rec_path, line))
                rec_loc = pointcloud_rec[:, :3]  # [numpoint, 3],location
                rec_color = np.array(rgb2yuv(pointcloud_rec[:, 3:])).astype(np.float32)  # color

                pointNum = rec_loc.shape[0]
                numPatch = pointNum * 2 // p_num
                pt_locs = np.expand_dims(rec_loc, 0)
                idx = farthest_point_sample(pt_locs, numPatch)
                centroid_xyz = rec_loc[idx, :]                    # [1, numPatch, 3]
                centroid_xyz = np.squeeze(centroid_xyz)           # [numPatch, 3]

                print("k nearest neighbor doing...")
                rec_loc_temp = torch.tensor(rec_loc).to(device)
                # print(rec_loc_temp.size())
                group_idx = torch.zeros(centroid_xyz.shape[0], p_num).to(device)

                patch_seg_time1 = time.time()
                for ii in range(numPatch):
                    centroid_xyz_temp = torch.unsqueeze(torch.tensor(centroid_xyz[ii, :]), dim=0).\
                        contiguous().to(device)
                    # print(torch.unsqueeze(torch.tensor(centroid_xyz[:,ii,:]).to(device),dim = -2).size())
                    group_idx_temp = search_knn(centroid_xyz_temp, rec_loc_temp, p_num)   # group_idx: [cen, k]
                    group_idx[ii, :] = group_idx_temp
                patch_seg_time2 = time.time()
                log_string(LOG_FOUT, "extract patches...%s" % datetime.datetime.now())
                new_color1 = torch.zeros(pointNum, dtype=torch.float32).to(device)
                new_color2 = torch.zeros(pointNum, dtype=torch.float32).to(device)
                new_color3 = torch.zeros(pointNum, dtype=torch.float32).to(device)
                c1_ind = torch.zeros(pointNum)

                datas1 = torch.zeros((args.test_batch_size, 4, p_num)).to(device)
                datas2 = torch.zeros((args.test_batch_size, 4, p_num)).to(device)
                datas3 = torch.zeros((args.test_batch_size, 4, p_num)).to(device)
                ji = 0
                process_time_sum = 0
                for j in range(numPatch):
                    process_time1 = time.time()
                    data_id = group_idx[j, :].long()
                    data_loc = torch.tensor(rec_loc)[data_id].to(device)
                    data_col = torch.tensor(rec_color)[data_id].to(device)

                    data1 = torch.cat((data_loc, torch.unsqueeze(data_col[:, 0], dim=-1)), dim=-1)
                    data1 = torch.unsqueeze(data1, dim=0)
                    data1 = data1.permute(0, 2, 1)
                    datas1[ji, :, :] = data1

                    data2 = torch.cat((data_loc, torch.unsqueeze(data_col[:, 1], dim=-1)), dim=-1)
                    data2 = torch.unsqueeze(data2, dim=0)
                    data2 = data2.permute(0, 2, 1)
                    datas2[ji, :, :] = data2

                    data3 = torch.cat((data_loc, torch.unsqueeze(data_col[:, 2], dim=-1)), dim=-1)
                    data3 = torch.unsqueeze(data3, dim=0)
                    data3 = data3.permute(0, 2, 1)
                    datas3[ji, :, :] = data3
                    ji = ji + 1
                    if ji < args.test_batch_size:
                        continue
                    ji = 0

                    with torch.no_grad():
                        logits1 = model1(datas1)           #[8, 2048, 1]
                        logits2 = model2(datas2)  # [8, 2048, 1]
                        logits3 = model3(datas3)  # [8, 2048, 1]

                    logits1 = torch.squeeze(logits1)
                    logits2 = torch.squeeze(logits2)
                    logits3 = torch.squeeze(logits3)
                    process_time2 = time.time()
                    process_time_sum += (process_time2 - process_time1)

                    for ii in range(args.test_batch_size):
                        idx_temps = group_idx[j - args.test_batch_size + 1 + ii, :].long()
                        logit1 = logits1[ii, :]
                        logit2 = logits2[ii, :]
                        logit3 = logits3[ii, :]
                        for t, m in enumerate(idx_temps):
                            new_color1[m] += logit1[t]
                            new_color2[m] += logit2[t]
                            new_color3[m] += logit3[t]
                            c1_ind[m] += 1

                print("patch_test done...")
                patch_fuse_time_beg = time.time()

                rec_color_ten = torch.tensor(rec_color)

                for pn in range(pointNum):
                    if c1_ind[pn] == 0:
                        new_color1[pn] = rec_color_ten[pn, 0]
                        new_color2[pn] = rec_color_ten[pn, 1]
                        new_color3[pn] = rec_color_ten[pn, 2]
                    elif c1_ind[pn] > 1:
                        new_color1[pn] /= c1_ind[pn]
                        new_color2[pn] /= c1_ind[pn]
                        new_color3[pn] /= c1_ind[pn]

                patch_fuse_time = time.time()
                output_color11 = np.array(torch.unsqueeze(new_color1, dim=-1).cpu())
                output_color12 = np.array(torch.unsqueeze(new_color2, dim=-1).cpu())
                output_color13 = np.array(torch.unsqueeze(new_color3, dim=-1).cpu())

                output_color = np.concatenate((output_color11, output_color12, output_color13), axis=-1)
                output_color = np.clip(np.round(yuv2rgb(output_color)), 0, 255)

                output = pointcloud_rec
                psnr_ori1 = psnr(ori_color[:, 0], rec_color[:, 0], MAX=255)
                psnr_pred1 = psnr(ori_color[:, 0], rgb2yuv(output_color)[:, 0], MAX=255)
                psnr_ori2 = psnr(ori_color[:, 1], rec_color[:, 1], MAX=255)
                psnr_pred2 = psnr(ori_color[:, 1], rgb2yuv(output_color)[:, 1], MAX=255)
                psnr_ori3 = psnr(ori_color[:, 2], rec_color[:, 2], MAX=255)
                psnr_pred3 = psnr(ori_color[:, 2], rgb2yuv(output_color)[:, 2], MAX=255)
                log_string(LOG_FOUT, "psnr_y for original:  %f" % psnr_ori1)
                log_string(LOG_FOUT, "psnr_y for pred:  %f" % psnr_pred1)
                log_string(LOG_FOUT, "psnr_u for original:  %f" % psnr_ori2)
                log_string(LOG_FOUT, "psnr_u for pred:  %f" % psnr_pred2)
                log_string(LOG_FOUT, "psnr_v for original:  %f" % psnr_ori3)
                log_string(LOG_FOUT, "psnr_v for pred:  %f" % psnr_pred3)
                output[:, 3:] = output_color
                filepath = os.path.join(args.pred_path, line)
                write_ply(output, filepath)

                # output_color = np.concatenate((output_color11, np.array(rec_color[:, 1:])), axis=-1)
                # output_color = np.clip(np.round(yuv2rgb(output_color)), 0, 255)
                # psnr_ori1 = psnr(ori_color[:, 0], rec_color[:, 0], MAX=255)
                # psnr_pred1 = psnr(ori_color[:, 0], rgb2yuv(output_color)[:, 0], MAX=255)
                # log_string(LOG_FOUT, "psnr_y for original:  %f" % psnr_ori1)
                # log_string(LOG_FOUT, "psnr_y for pred:  %f" % psnr_pred1)
                #
                # output_color = np.concatenate((np.array(np.expand_dims(rec_color[:, 0], axis=-1)), output_color12,
                #                                np.array(np.expand_dims(rec_color[:, 2], axis=-1))), axis=-1)
                # output_color = np.clip(np.round(yuv2rgb(output_color)), 0, 255)
                # psnr_ori2 = psnr(ori_color[:, 1], rec_color[:, 1], MAX=255)
                # psnr_pred2 = psnr(ori_color[:, 1], rgb2yuv(output_color)[:, 1], MAX=255)
                # log_string(LOG_FOUT, "psnr_u for original:  %f" % psnr_ori2)
                # log_string(LOG_FOUT, "psnr_u for pred:  %f" % psnr_pred2)
                #
                # output_color = np.concatenate((np.array(rec_color[:, :2]), output_color13), axis=-1)
                # output_color = np.clip(np.round(yuv2rgb(output_color)), 0, 255)
                # psnr_ori3 = psnr(ori_color[:, 2], rec_color[:, 2], MAX=255)
                # psnr_pred3 = psnr(ori_color[:, 2], rgb2yuv(output_color)[:, 2], MAX=255)
                # log_string(LOG_FOUT, "psnr_v for original:  %f" % psnr_ori3)
                # log_string(LOG_FOUT, "psnr_v for pred:  %f \n" % psnr_pred3)

                log_string(LOG_FOUT, "patch_seg time:  %f" % (patch_seg_time2 - patch_seg_time1))
                log_string(LOG_FOUT, "processing time:  %f" % process_time_sum)
                log_string(LOG_FOUT, "patch_fuse time:  %f \n \n" % (patch_fuse_time - patch_fuse_time_beg))

    LOG_FOUT.close()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud quality Enhancement')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--pth_path', type=str, default='pths/final_2023')
    parser.add_argument('--log_path', type=str, default='logs/final_2023')
    parser.add_argument('--log_path_test', type=str, default='logs_test_2023/final_2023')
    parser.add_argument('--train_h5_txt', type=str, default='data_LT/h5_mix_new/trainFile.txt')
    parser.add_argument('--test_ply_txt', type=str, default='data_LT/data_ori_add/testFile.txt')
    parser.add_argument('--test_ori_ply', type=str, default='data_LT/data_ori_add/same_order/')
    parser.add_argument('--test_rec_ply', type=str, default='data_LT/data_rec_add/')
    parser.add_argument('--train_channel', type=int, default=0, help='0:Y, 1:U, 2:V')
    parser.add_argument('--valid_h5_txt', type=str, default='data_LT/h5_mix_test_new/testFile.txt')
    parser.add_argument('--model', type=str, default='GQE-Net', metavar='N',
                        help='Model to use, GQE-Net')
    parser.add_argument('--dataset', type=str, default='WPCSD', metavar='N',
                        choices=['WPCSD'])
    parser.add_argument('--batch_size', type=int, default=12, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='test_batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=180, metavar='N',
                        help='Number of episode to train ')
    parser.add_argument('--bit_rate_point', type=str, default='r01_r06_yuv_GQE-Net')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--has_model', type=bool, default=False,
                        help='Checkpoints')
    parser.add_argument('--lr', type=float, default=0.0025, metavar='LR',
                        help='Learning rate (default: 0.005, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='Enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='Random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='Evaluate the model (Test stage)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Num of points to use')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model1_path', type=str, default='pths/final_2023/GQE-Net/2023-07-25/y/model_6.pth', metavar='N',
                        help='Pretrained model1 path')
    parser.add_argument('--model2_path', type=str, default='pths/final_2023/GQE-Net/2023-07-28/u/model_55.pth', metavar='N',
                        help='Pretrained model2 path')
    parser.add_argument('--model3_path', type=str, default='pths/final_2023/GQE-Net/2023-07-31/v/model_92.pth', metavar='N',
                        help='Pretrained model3 path')
    parser.add_argument('--pred_path', type=str, default='data/preds/final_2023')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        torch.cuda.set_device(devices)
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')

        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
