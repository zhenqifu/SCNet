
import os
import torch
from torch.utils.data import DataLoader
from net_wosn import Net
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from data import get_training_set, get_eval_set
import socket
import time
from utils import quantize, calc_psnr_ssim

# Training settings
parser = argparse.ArgumentParser(description='PyTorch SCNet')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=256, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='100000', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--device', type=str, default='cuda:3')
parser.add_argument('--train_data_dir', type=str, default='../Dataset/UIE/UIEBD/train/image')
parser.add_argument('--train_label_dir', type=str, default='../Dataset/UIE/UIEBD/train/label')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--eval_data_dir', type=str, default='../Dataset/UIE/UIEBD/test/image')
parser.add_argument('--eval_label_dir', type=str, default='../Dataset/UIE/UIEBD/test/label')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped HR image')
parser.add_argument('--model', default='none', help='Pretrained model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')


opt = parser.parse_args()
device = torch.device(opt.device)
hostname = str(socket.gethostname())
cudnn.benchmark = True

best_psnr = 0
best_epoch_psnr = 0
best_ssim = 0
best_epoch_ssim = 0


def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        Input, target = batch[0], batch[1]
        if cuda:
            Input = Input.to(device)
            target = target.to(device)

        t0 = time.time()
        out = model.forward(Input)
        loss = model.loss(out, target)
        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        t1 = time.time()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={} || Timer: {:.4f} sec.".format(epoch, iteration, 
                          len(training_data_loader), loss.item(), optimizer.param_groups[0]['lr'], (t1 - t0)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test(testing_data_loader):
    global best_psnr
    global best_epoch_psnr
    global best_ssim
    global best_epoch_ssim

    avg_psnr = 0
    avg_ssim = 0

    torch.set_grad_enabled(False)
    epoch = scheduler.last_epoch
    
    model.eval()
    
    print('\nEvaluation:')
    
    for batch in testing_data_loader:
        with torch.no_grad():
            Input, target, name = batch[0], batch[1], batch[2]
        if cuda:
            Input = Input.to(device)
            target = target.to(device)
            
            with torch.no_grad():
                out = model.forward(Input)
                prediction = quantize(out, opt.rgb_range)
                target = quantize(target, opt.rgb_range)
                psnr, ssim = calc_psnr_ssim(prediction, target)
                avg_psnr += psnr
                avg_ssim += ssim

    avg_psnr = avg_psnr / len(testing_data_loader)
    avg_ssim = avg_ssim / len(testing_data_loader)
    checkpoint(epoch)

    if avg_psnr >= best_psnr:
        # checkpoint(epoch)
        best_epoch_psnr = epoch
        best_psnr = avg_psnr
    if avg_ssim >= best_ssim:
        best_epoch_ssim = epoch
        best_ssim = avg_ssim
    print("===> Avg.PSNR: {:.4f} dB ||  Best.PSNR: {:.4f} dB || Epoch: {}".format(avg_psnr, best_psnr, best_epoch_psnr))
    print("===> Avg.SSIM: {:.4f} dB ||  Best.SSIM: {:.4f} dB || Epoch: {}".format(avg_ssim, best_ssim, best_epoch_ssim))
    torch.set_grad_enabled(True)


def checkpoint(epoch):
    model_out_path = opt.save_folder+"epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')

test_set = get_eval_set(opt.eval_data_dir, opt.eval_label_dir)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

train_set = get_training_set(opt.train_data_dir, opt.train_label_dir, opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

model = Net(opt)

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_model)
    if os.path.exists(model_name):
        # model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained model is loaded.')

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

milestones = []
for i in range(1, opt.nEpochs+1):
    if i % opt.decay == 0:
        milestones.append(i)

        
scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)
                            
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    
    train(epoch)
    scheduler.step()

    if (epoch+1) % opt.snapshots == 0:
        test(testing_data_loader)


