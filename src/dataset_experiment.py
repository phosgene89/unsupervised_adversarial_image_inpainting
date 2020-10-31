import os
from collections import Mapping
import torch
from ignite._utils import convert_tensor
from collections import OrderedDict
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import torch
from scipy.stats import truncnorm
from torch import nn
from torchvision.utils import make_grid

from src.models.GD_factory import G_D
from src.models.model_utils.losses import GANLoss
from src.fid.inception import InceptionV3
from src.models.model_utils.net_utils import Distribution, set_requires_grad, mod_to_gpu, fix_seed

class BaseExperiment(object):
    def __init__(self, seed=None, **kwargs):
        self._modules = OrderedDict()
        self._optimizers = OrderedDict()
        self._datasets = OrderedDict()

        if seed is not None:
            fix_seed(seed)

    def evaluating(self):
        self.training(mode=False)

    def zero_grad(self):
        for optim in self.optimizers():
            optim.zero_grad()

    def to(self, device):
        for m in self.modules():
            m.to(device)
        return self

# Implement creation of root directory.

class DatasetExperiment(BaseExperiment):
    def __init__(self, train, niter, nepoch, eval=None, gpu_id=[0],
                 root=None, corruption=None, device="cpu",
                 **kwargs):
        super(DatasetExperiment, self).__init__(**kwargs)

        self.train = train
        self.eval = eval

        self.corruption = corruption

        self.niter = niter
        self.nepoch = nepoch
        self.device = device

        self.gpu_id = gpu_id

class GAN_experiment(DatasetExperiment):
    def __init__(self, gen, dis, optim_gen, optim_dis, nz, random_type='normal', fid=True, num_samples=8, truncation=0,
                 fp16=False, G_batch_size=16, num_D_step=1, num_D_acc=1, num_G_acc=1, gan_mode='hinge',
                 **kwargs):
        super(GAN_experiment, self).__init__(**kwargs)

        self.nz = nz
        self.random_type = random_type
        self.num_D_step = num_D_step
        self.num_D_acc = num_D_acc
        self.num_G_acc = num_G_acc

        self.truncation = truncation

        self.num_samples = num_samples

        self.gen = gen.to(self.device)
        self.dis = dis.to(self.device)

        self.optim_gen = optim_gen
        self.optim_dis = optim_dis

        self.fp16 = fp16


        G_batch_size = self.train.batch_size
        self.z = Distribution(torch.randn(G_batch_size, self.nz, requires_grad=False))
        self.z.init_distribution(dist_type=random_type, mean=0, var=1)
        self.z = self.z.to(self.device, torch.float16 if fp16 else torch.float32)
        if fp16:
            self.z = self.z.half()

        if self.fp16:
            self.gen = self.gen.half()
            self.dis = self.dis.half()

        self.GD = G_D(self.gen, self.dis)
        self.gan_loss = GANLoss(gan_mode=gan_mode)

        if type(self.gpu_id) == int:
            self.gpu_id = [self.gpu_id]
        if len(self.gpu_id) > 1:
            assert (torch.cuda.is_available())
            self.GD = torch.nn.DataParallel(self.GD, self.gpu_id).cuda()  # multi-GPUs
        else:
            self.GD = self.GD.to(self.device)

        self.fid = fid
        if self.fid:
            self.max_fid = float('inf')

            self.dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
            self.model = InceptionV3([block_idx])
            self.fid_score = float('inf')
            if len(self.gpu_id) > 1:
                assert (torch.cuda.is_available())
                self.model = torch.nn.DataParallel(self.model, self.gpu_id).cuda()  # multi-GPUs
            else:
                self.model = self.model.to(self.device)



    def optimize(self, x, label, **kwargs):
        self.optim_gen.zero_grad()
        self.optim_dis.zero_grad()

        set_requires_grad(self.dis, True)
        set_requires_grad(self.gen, False)

        if self.fp16:
            x = x.half()

        for step_index in range(self.num_D_step):
            for accumulation_index in range(self.num_D_acc):
                self.z.sample_()

                pred_real, pred_fake = self.GD(z=self.z, x=x, label=label, train_G=False)
                loss_dis_real, loss_dis_fake = self.gan_loss.D_loss(pred_fake, pred_real)
                loss_dis = (loss_dis_real + loss_dis_fake) / float(self.num_D_acc)
                loss_dis.backward()
            self.optim_dis.step()

        set_requires_grad(self.dis, False)
        set_requires_grad(self.gen, True)

        self.optim_gen.zero_grad()
        for accumulation_index in range(self.num_G_acc):
            self.z.sample_()
            pred_fake = self.GD(z=self.z, x=None, label=label, train_G=True)
            loss_gen = self.gan_loss.G_loss(pred_fake)
            loss_gen = loss_gen / float(self.num_G_acc)
            loss_gen.backward()
        self.optim_gen.step()

        return dict(loss_gen=loss_gen, loss_dis_real=loss_dis_real, loss_dis_fake=loss_dis_fake)


    def samples(self, num_samples=12):
        ims = []

        z_dist = Distribution(torch.randn(num_samples, self.nz, requires_grad=False))
        z_dist.init_distribution(dist_type=self.random_type, mean=0, var=1)
        z_dist = z_dist.to(self.device, torch.float16 if self.fp16 else torch.float32)
        if self.fp16:
            z_dist = z_dist.half()

        # for j in range(sample_per_sheet):
        # batch = convert_tensor(batch, self.device)
        z_dist.sample_()
        with torch.no_grad():
            if self.truncation > 0:
                z = self.truncated_z_sample(batch_size=num_samples, truncation=self.truncation).to(self.device)
                o = self.gen(z=z)
            else:
                self.z.sample_()
                o = self.gen(z=z_dist)

            if self.fp16:
                o = o.float()

        out_ims = o
        return o

    def interp(self, num_samples=12, fix=True):

        def interpolation(x0, x1, num_midpoints):
            lerp = torch.linspace(0, 1.0, num_midpoints + 2, device='cuda').to(x0.dtype)
            return ((x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1)))

        if self.truncation > 0:
            zs = interpolation(self.truncated_z_sample(batch_size=1, truncation=self.truncation).to(self.device),
                               self.truncated_z_sample(batch_size=1, truncation=self.truncation).to(self.device),
                               num_samples - 2).view(-1, self.nz)

        else:
            zs = interpolation(torch.randn(1, 1, self.nz, device=self.device),
                               torch.randn(1, 1, self.nz, device=self.device),
                               num_samples - 2).view(-1, self.nz)

        with torch.no_grad():
            if self.fp16:
                zs = zs.half()
            # if self.truncation > 0:
            o = self.gen(z=zs)
            if self.fp16:
                o = o.float()

        out_ims = o
        return o

    def compute_fid(self, iteration):
        self.evaluating()
        fake_list, real_list = [], []
        with torch.no_grad():
            for i, batch in enumerate(self.eval):
                if self.truncation > 0:
                    z = self.truncated_z_sample(batch_size=batch['x'].size(0), truncation=self.truncation).to(self.device)
                    fake = self.gen(z=z)
                else:
                    self.z.sample_()
                    fake = self.gen(z=self.z)
                true = batch['x']
                if self.fp16:
                    true = true.float()
                    fake = fake.float()

                fake_list.append((fake.cpu() + 1.0) / 2.0)
                real_list.append((true.cpu() + 1.0) / 2.0)

        fake_images = torch.cat(fake_list)
        real_images = torch.cat(real_list)

    def truncated_z_sample(self, batch_size, truncation=1., seed=None):
        values = truncnorm.rvs(-2, 2, size=(batch_size, self.nz))
        return torch.tensor(truncation * values).float()
    
# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D_E(nn.Module):
    def __init__(self, G, D, E, corruption, lambda_z=1, lambda_L1=1, lambda_gen_2=0):
        super(G_D_E, self).__init__()
        self.G = G
        self.D = D
        self.E = E
        self.corruption = corruption
        self.lambda_z = lambda_z
        self.lambda_L1 = lambda_L1
        self.lambda_gen_2 = lambda_gen_2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, y, x=None, z=None, theta=None, label=None, train_G=False):
        # If training G, enable grad tape

        with torch.set_grad_enabled(train_G):
            # Get Generator output given noise
            x_hat_orig = self.G(y, z, label)
            # Line below uses 1-theta to invert mask. Seems to be done because original random mask
            # masked few values and this is an easy way to make them mask more values.
            #x_hat_mask, _ = self.corruption(x_hat_orig, theta=1 - theta) original line...might need to swap mask and _ however...
            x_hat, x_hat_mask = self.corruption(x_hat_orig, theta=theta)
            print("LOOK HERE: {}, {}, {}".format(x_hat.type(), theta.type(), y.type()))
            x_hat = x_hat*theta.long().to(self.device) + y * (~theta).long().to(self.device)
            y_tilde, theta2 = self.corruption(x_hat)

        if train_G:
            if self.lambda_z > 0:
                z_hat = self.E(x_hat)
            else:
                z_hat = None
            if self.lambda_L1 > 0.0 or self.lambda_gen_2 > 0:
                z_recon = self.E(y)
                x_recon = self.G(y_tilde, z_recon, label=label)
                y_recon, _ = self.corruption(x_recon, theta=theta)
                ## *******************************************
                x_recon_sample = x_recon[0].clone().detach().cpu().numpy().swapaxes(0,2).swapaxes(0,1)
                x_recon_sample = 0.5 * x_recon_sample + 0.5
                
                x_hat_sample = x_hat[0].clone().detach().cpu().numpy().swapaxes(0,2).swapaxes(0,1)
                x_hat_sample = 0.5 * x_hat_sample + 0.5
                
                y_recon_sample = y_recon[0].detach().clone().detach().cpu().numpy().swapaxes(0,2).swapaxes(0,1)
                y_recon_sample = 0.5 * y_recon_sample + 0.5
                
                y_tilde_sample = y_tilde[0].clone().detach().cpu().numpy().swapaxes(0,2).swapaxes(0,1)
                y_tilde_sample = 0.5 * y_tilde_sample + 0.5
                
                y_sample = y[0].detach().clone().detach().cpu().numpy().swapaxes(0,2).swapaxes(0,1)
                y_sample = 0.5 * y_sample + 0.5
                
                fig, axs = plt.subplots(1,5, figsize=(15, 5))
                
                if np.random.uniform() < 1:
                
                    axs[0].imshow(x_recon_sample)
                    axs[0].set_title("x_reconstruction")
                    axs[1].imshow(y_recon_sample)
                    axs[1].set_title("y_reconstruction")
                    axs[2].imshow(y_tilde_sample)
                    axs[2].set_title("y_tilde")
                    axs[3].imshow(x_hat_sample)
                    axs[3].set_title("x_hat")
                    axs[4].imshow(y_sample)
                    axs[4].set_title("y")
                    plt.show()
                ## *******************************************
                if self.lambda_gen_2:
                    pred_fake_recon = self.D(y_recon, label)
                else:
                    pred_fake_recon = None
            else:
                pred_fake_recon = None
                y_recon = None

            pred_fake = self.D(y_tilde, label)

            return pred_fake, pred_fake_recon, y_recon, z_hat, x_hat, y_tilde
        else:
            pred_real = self.D(y, label)
            pred_fake = self.D(y_tilde.detach(), label)

            return pred_real, pred_fake


class UnsupervisedImageInpainting(GAN_experiment):
    def __init__(self, lambda_z=1, lambda_kl=1, lambda_L1=1, lambda_gen_2=0, use_l1=True, enc=None, **kwargs):
        super(UnsupervisedImageInpainting, self).__init__(**kwargs)

        self.lambda_z = lambda_z
        self.lambda_kl = lambda_kl
        self.lambda_L1 = lambda_L1
        self.lmbda_gen_2 = lambda_gen_2

        # self.enc = mod_to_gpu(enc, self.gpu_id, self.device)

        self.GDE = G_D_E(self.gen, self.dis, enc, corruption=self.corruption, lambda_z=self.lambda_z, lambda_L1=self.lambda_L1)
        self.GDE = mod_to_gpu(self.GDE, self.gpu_id, self.device)
        self.criterionL2 = nn.MSELoss()
        self.criterionL1 = nn.L1Loss()

        if use_l1:
            self.recon_loss_func = self.criterionL1
        else:
            self.recon_loss_func = self.criterionL2

    def optimize(self, x, y, theta, label, **kwargs):

        self.y, self.x = y, x

        self.optim_gen.zero_grad()
        self.optim_dis.zero_grad()
        set_requires_grad(self.dis, True)
        set_requires_grad(self.gen, False)

        if self.fp16:
            self.x_a = self.x_a.half()

        for step_index in range(self.num_D_step):
            for accumulation_index in range(self.num_D_acc):
                self.z.sample_()
                pred_real, pred_fake = self.GDE(y=self.y, x=self.x, z=self.z, theta=theta, label=label,
                                                train_G=False)
                loss_dis_real, loss_dis_fake = self.gan_loss.D_loss(pred_fake, pred_real)
                loss_dis = (loss_dis_real + loss_dis_fake) / float(self.num_D_acc)
                loss_dis.backward()
            self.optim_dis.step()

        set_requires_grad(self.dis, False)
        set_requires_grad(self.gen, True)

        self.optim_gen.zero_grad()
        for accumulation_index in range(self.num_G_acc):
            self.z.sample_()
            pred_fake, pred_fake_recon, self.y_recon, z_hat, self.x_hat, self.y_tilde = self.GDE(y=self.y,
                                                                                                 x=None,
                                                                                                 z=self.z,
                                                                                                 theta=theta,
                                                                                                 label=label,
                                                                                                 train_G=True)
            loss_gen_gan = self.gan_loss.G_loss(pred_fake)
            if self.lmbda_gen_2 > 0:
                loss_gen_gan_recon = self.gan_loss.G_loss(pred_fake_recon)
            else:
                loss_gen_gan_recon = 0

            if self.lambda_L1 > 0:
                loss_gen_l1 = self.criterionL1(self.y_recon, self.y) * self.lambda_L1
            else:
                loss_gen_l1 = torch.tensor(0)
            if self.lambda_z > 0:
                loss_gen_z = self.criterionL1(z_hat, self.z) * self.lambda_z
            else:
                loss_gen_z = torch.tensor(0)

            loss_gen = (loss_gen_gan + loss_gen_gan_recon + loss_gen_l1 + loss_gen_z) / float(self.num_G_acc)
            loss_gen.backward()
        self.optim_gen.step()

        if x is not None:
            recon = self.criterionL2(self.x_hat, x)
            
        else:
            recon = 0

        return dict(loss_gen=loss_gen,
                    l1_z= loss_gen_l1 / max(0.01, self.lambda_L1),
                    l1_Y= loss_gen_z / max(0.01, self.lambda_z),
                    loss_gen_l1=loss_gen_l1,
                    loss_dis_real=loss_dis_real,
                    recon=recon,
                    loss_dis_fake=loss_dis_fake)

    def write_image(self, engine, dataset_name):
        iteration = self.trainer.state.iteration

        b = engine.state.batch

        x = b['x'].cpu()
        y = b['y'].cpu()
        perm = torch.randperm(x.size(0))
        idx = perm[:8]
        list_tensor = []

        x = x[idx]
        y = y[idx]
        list_tensor.append(x.cpu())
        list_tensor.append(y.cpu())

        x_hat = self.x_hat[idx].cpu()
        list_tensor.append(x_hat)

        y_tilde = self.y_tilde[idx].cpu()
        list_tensor.append(y_tilde)
        if self.lambda_L1 > 0:
            y_recon = self.y_recon[idx].cpu()
            list_tensor.append(y_recon)
        img_tensor = torch.cat(list_tensor, dim=0)

        img = make_grid(
                img_tensor,
                nrow=self.num_samples, scale_each=True, normalize=True,
                )

        try:
            self.writers.add_image(dataset_name, img, iteration)
        except:
            self.pbar.log_message('IMPOSSIBLE TO SAVE')

    def compute_fid(self, iteration):
        self.evaluating()
        fake_list, real_list = [], []
        with torch.no_grad():
            for i, batch in enumerate(self.eval):
                true = batch['x'].cuda()
                if self.nz > 0:
                    self.z.sample_()
                    z = self.z
                else:
                    z = None
                fake = self.gen(x=batch['y'].cuda(), z=z)
                if self.fp16:
                    true = true.float()
                    fake = fake.float()

                fake_list.append((fake.cpu() + 1.0) / 2.0)
                real_list.append((true.cpu() + 1.0) / 2.0)

        fake_images = torch.cat(fake_list)
        real_images = torch.cat(real_list)
        mu_fake, sigma_fake = metrics.calculate_activation_statistics(
                fake_images, self.model, self.train.batch_size, device=self.device
                )
        mu_real, sigma_real = metrics.calculate_activation_statistics(
                real_images, self.model, self.train.batch_size, device=self.device
                )
        self.fid_score = metrics.calculate_frechet_distance(
                mu_fake, sigma_fake, mu_real, sigma_real
                )

        if self.writers is not None:
            self.writers.add_scalar('FID', self.fid_score, iteration)
