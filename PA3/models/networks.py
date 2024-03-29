import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.optim import lr_scheduler
from models.normalize import RAIN
from torch.nn.utils import spectral_norm

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    elif norm_type.startswith('rain'):
        norm_layer = RAIN
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False,
             init_type='normal', init_gain=0.02, gpu_ids=[]):
    """load a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: rainnet
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'rainnet':
        net = RainNet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_attention=True)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
            self.relu = nn.ReLU()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean() # self.relu(1-prediction.mean())
            else:
                loss = prediction.mean() # self.relu(1+prediction.mean())
        return loss

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0, mask=None):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.unsqueeze(2).unsqueeze(3)
            alpha = alpha.expand_as(real_data)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv, mask, gp=True)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True,
                                        allow_unused=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

def get_act_conv(act, dims_in, dims_out, kernel, stride, padding, bias):
    conv = [act]
    conv.append(nn.Conv2d(dims_in, dims_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias))
    return nn.Sequential(*conv)

def get_act_dconv(act, dims_in, dims_out, kernel, stride, padding, bias):
    conv = [act]
    conv.append(nn.ConvTranspose2d(dims_in, dims_out, kernel_size=kernel, stride=2, padding=1, bias=False))
    return nn.Sequential(*conv)

class RainNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=RAIN,
                 norm_type_indicator=[0, 0, 0, 0, 0, 0, 0, 1,1,1,1,1,1,1],
                 use_dropout=False, use_attention=True):
        super(RainNet, self).__init__()
        self.input_nc = input_nc
        self.norm_namebuffer = ['RAIN']
        self.use_dropout = use_dropout
        self.use_attention = use_attention
        norm_type_list = [get_norm_layer('instance'), get_norm_layer('rain')]
        # -------------------------------Network Settings-------------------------------------
        self.act2=nn.ReLU()
        self.layer0=nn.Conv2d(input_nc,ngf,4,2,1)

        self.act1=nn.LeakyReLU()
        self.layer1=nn.Conv2d(ngf,ngf*2,4,2,1)
        self.norm_layer1=norm_type_list[norm_type_indicator[0]](2*ngf)

        self.layer2=nn.Conv2d(ngf*2,ngf*4,4,2,1)
        self.norm_layer2=norm_type_list[norm_type_indicator[1]](4*ngf)

        self.layer3=nn.Conv2d(ngf*4,ngf*8,4,2,1)
        self.norm_layer3=norm_type_list[norm_type_indicator[2]](8*ngf)

        self.layer4=nn.Conv2d(ngf*8,ngf*8,4,2,1)
        self.norm_layer4=norm_type_list[norm_type_indicator[3]](8*ngf)

        self.layer5=nn.Conv2d(ngf*8,ngf*8,4,2,1)
        self.norm_layer5=norm_type_list[norm_type_indicator[4]](8*ngf)

        self.layer6=nn.Conv2d(ngf*8,ngf*8,4,2,1)
        self.norm_layer6=norm_type_list[norm_type_indicator[5]](8*ngf)

        self.layer7=nn.Conv2d(ngf*8,ngf*8,4,2,1)
        self.norm_layer7=norm_type_list[norm_type_indicator[6]](8*ngf)
        
        self.layer8=nn.ConvTranspose2d(ngf*8,ngf*8,4,2,1)
        self.norm_layer8=norm_type_list[norm_type_indicator[7]](8*ngf)

        self.layer9=nn.ConvTranspose2d(ngf*16,ngf*8,4,2,1)
        self.norm_layer9=norm_type_list[norm_type_indicator[8]](8*ngf)

        self.layer10=nn.ConvTranspose2d(ngf*16,ngf*8,4,2,1)
        self.norm_layer10=norm_type_list[norm_type_indicator[9]](8*ngf)

        self.layer11=nn.ConvTranspose2d(ngf*16,ngf*8,4,2,1)
        self.norm_layer11=norm_type_list[norm_type_indicator[10]](8*ngf)

        self.layer12=nn.ConvTranspose2d(ngf*16,ngf*4,4,2,1)
        self.norm_layer12=norm_type_list[norm_type_indicator[11]](4*ngf)

        self.layer13=nn.ConvTranspose2d(ngf*8,ngf*2,4,2,1)
        self.norm_layer13=norm_type_list[norm_type_indicator[12]](2*ngf)

        self.layer14=nn.ConvTranspose2d(ngf*4,ngf,4,2,1)
        self.norm_layer14=norm_type_list[norm_type_indicator[13]](ngf)

        self.layer15=nn.ConvTranspose2d(2*ngf,output_nc,4,2,1)
        self.tanh_15=nn.Tanh()

        self.attention1=nn.Conv2d(8*ngf,8*ngf,1,1,0)
        self.attention1_sigmoid=nn.Sigmoid()

        self.attention2=nn.Conv2d(4*ngf,4*ngf,1,1,0)

        self.attention3=nn.Conv2d(2*ngf,2*ngf,1,1,0)


        # fill the blank


    def forward(self, x, mask):
        # fill the blank
        ######################################################################################
        #이 코드는 step 2에 대한 코드이다.                                                        #
        #이를 step 1을 확인하기 위해 바꾸려면 line 392, 398, 404, 410, 416, 426, 436 의 주석을 없애고 # 
        #line 393, 399, 405, 411, 417, 427, 437을 주석처리한 후                                 #
        #line 284의 norm_type_indicator의 모든 항을 0으로 바꾼 후 실행시키면 된다.                   #
        ######################################################################################
        x0=self.layer0(x)
        
        x1=self.act1(x0)
        x1=self.layer1(x1)
        x1=self.norm_layer1(x1)

        x2=self.act1(x1)
        x2=self.layer2(x2)
        x2=self.norm_layer2(x2)

        x3=self.act1(x2)
        x3=self.layer3(x3)
        x3=self.norm_layer3(x3)
        
        x4=self.act1(x3)
        x4=self.layer4(x4)
        x4=self.norm_layer4(x4)
        
        x5=self.act1(x4)
        x5=self.layer5(x5)
        x5=self.norm_layer5(x5)
        
        x6=self.act1(x5)
        x6=self.layer6(x6)
        x6=self.norm_layer6(x6)
        
        x7=self.act1(x6)
        x7=self.layer7(x7)

        out=self.act2(x7)
        out=self.layer8(out)
        # out=self.norm_layer8(out)
        out=self.norm_layer8(out,mask)
        out=torch.cat((out,x6),1)
        
        out=self.act2(out)
        out=self.layer9(out)
        # out=self.norm_layer9(out)
        out=self.norm_layer9(out,mask)
        out=torch.cat((out,x5),1)

        out=self.act2(out)
        out=self.layer10(out)
        # out=self.norm_layer10(out)
        out=self.norm_layer10(out,mask)
        out=torch.cat((out,x4),1)

        out=self.act2(out)
        out=self.layer11(out)
        # out=self.norm_layer11(out)
        out=self.norm_layer11(out,mask)
        out=torch.cat((out,x3),1)

        out=self.act2(out)
        out=self.layer12(out)
        # out=self.norm_layer12(out)
        out=self.norm_layer12(out,mask)
        out=torch.cat((out,x2),1)

        tmp=self.attention1(out)
        tmp=self.attention1_sigmoid(tmp)
        out=tmp*out

        out=self.act2(out)
        out=self.layer13(out)
        # out=self.norm_layer13(out)
        out=self.norm_layer13(out,mask)
        out=torch.cat((out,x1),1)

        tmp=self.attention2(out)
        tmp=self.attention1_sigmoid(tmp)
        out=tmp*out

        out=self.act2(out)
        out=self.layer14(out)
        # out=self.norm_layer14(out)
        out=self.norm_layer14(out,mask)
        out=torch.cat((out,x0),1)

        tmp=self.attention3(out)
        tmp=self.attention1_sigmoid(tmp)
        out=tmp*out

        out=self.layer15(out)
        out=self.tanh_15(out)
        
        return out

    def processImage(self, x, mask, background=None):
        if background is not None:
            x = x * mask + background * (1 - mask)
        if self.input_nc == 4:
            x = torch.cat([x, mask], dim=1)  # (bs, 4, 256, 256)
        pred = self.forward(x, mask)
        return pred * mask + x[:, :3, :, :] * (1 - mask)


class UnetBlockCodec(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=RAIN, use_dropout=False, use_attention=False, enc=True, dec=True):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetBlockCodec) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
            enc (bool) -- if use give norm_layer in encoder part.
            dec (bool) -- if use give norm_layer in decoder part.
        """
        super(UnetBlockCodec, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.use_dropout = use_dropout
        self.use_attention = use_attention
        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        self.norm_namebuffer = ['RAIN', 'RAIN_Method_Learnable', 'RAIN_Method_BN']
        if outermost:
            self.down = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            self.submodule = submodule
            self.up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
        elif innermost:
            self.up = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
                nn.ReLU(True),
                nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            )
            self.upnorm = norm_layer(outer_nc) if dec else get_norm_layer('instance')(outer_nc)
        else:
            self.down = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
            )
            self.downnorm = norm_layer(inner_nc) if enc else get_norm_layer('instance')(inner_nc)
            self.submodule = submodule
            self.up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
            )
            self.upnorm = norm_layer(outer_nc) if dec else get_norm_layer('instance')(outer_nc)
            if use_dropout:
                self.dropout = nn.Dropout(0.5)

        if use_attention:
            attention_conv = nn.Conv2d(outer_nc+input_nc, outer_nc+input_nc, kernel_size=1)
            attention_sigmoid = nn.Sigmoid()
            self.attention = nn.Sequential(*[attention_conv, attention_sigmoid])

    def forward(self, x, mask):
        if self.outermost:
            x = self.down(x)
            x = self.submodule(x, mask)
            ret = self.up(x)
            return ret
        elif self.innermost:
            ret = self.up(x)
            if self.upnorm._get_name() in self.norm_namebuffer:
                ret = self.upnorm(ret, mask)
            else:
                ret = self.upnorm(ret)
            ret = torch.cat([x, ret], 1)
            if self.use_attention:
                return self.attention(ret) * ret
            return ret
        else:
            ret = self.down(x)
            if self.downnorm._get_name() in self.norm_namebuffer:
                ret = self.downnorm(ret, mask)
            else:
                ret = self.downnorm(ret)
            ret = self.submodule(ret, mask)
            ret = self.up(ret)
            if self.upnorm._get_name() in self.norm_namebuffer:
                ret = self.upnorm(ret, mask)
            else:
                ret = self.upnorm(ret)
            if self.use_dropout:    # only works for middle features
                ret = self.dropout(ret)
            ret = torch.cat([x, ret], 1)
            if self.use_attention:
                return self.attention(ret) * ret
            return ret


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        self.return_mask = True

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class OrgDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=6, norm_layer=nn.BatchNorm2d, global_stages=0):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(OrgDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 3
        padw = 0
        self.conv1 = spectral_norm(PartialConv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw))
        if global_stages < 1:
            self.conv1f = spectral_norm(PartialConv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw))
        else:
            self.conv1f = self.conv1
        self.relu1 = nn.LeakyReLU(0.2, True)
        nf_mult = 1
        nf_mult_prev = 1

        n = 1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv2 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm2 = norm_layer(ndf * nf_mult)
        if global_stages < 2:
            self.conv2f = spectral_norm(
                PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
            self.norm2f = norm_layer(ndf * nf_mult)
        else:
            self.conv2f = self.conv2
            self.norm2f = self.norm2

        self.relu2 = nn.LeakyReLU(0.2, True)

        n = 2
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv3 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm3 = norm_layer(ndf * nf_mult)
        if global_stages < 3:
            self.conv3f = spectral_norm(
                PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
            self.norm3f = norm_layer(ndf * nf_mult)
        else:
            self.conv3f = self.conv3
            self.norm3f = self.norm3
        self.relu3 = nn.LeakyReLU(0.2, True)

        n = 3
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.norm4 = norm_layer(ndf * nf_mult)
        self.conv4 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.conv4f = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm4f = norm_layer(ndf * nf_mult)

        self.relu4 = nn.LeakyReLU(0.2, True)

        n = 4
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv5 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.conv5f = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm5 = norm_layer(ndf * nf_mult)
        self.norm5f = norm_layer(ndf * nf_mult)
        self.relu5 = nn.LeakyReLU(0.2, True)

        n = 5
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv6 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.conv6f = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm6 = norm_layer(ndf * nf_mult)
        self.norm6f = norm_layer(ndf * nf_mult)
        self.relu6 = nn.LeakyReLU(0.2, True)

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.conv7 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias))
        self.conv7f = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias))

    def forward(self, input, mask=None):
        x = input
        x, _ = self.conv1(x)
        x = self.relu1(x)
        x, _ = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x, _ = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x, _ = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)
        x, _ = self.conv5(x)
        x = self.norm5(x)
        x = self.relu5(x)
        x, _ = self.conv6(x)
        x = self.norm6(x)
        x = self.relu6(x)
        x, _ = self.conv7(x)

        """Standard forward."""
        xf, xb = input, input
        mf, mb = mask, 1 - mask

        xf, mf = self.conv1f(xf, mf)
        xf = self.relu1(xf)
        xf, mf = self.conv2f(xf, mf)
        xf = self.norm2f(xf)
        xf = self.relu2(xf)
        xf, mf = self.conv3f(xf, mf)
        xf = self.norm3f(xf)
        xf = self.relu3(xf)
        xf, mf = self.conv4f(xf, mf)
        xf = self.norm4f(xf)
        xf = self.relu4(xf)
        xf, mf = self.conv5f(xf, mf)
        xf = self.norm5f(xf)
        xf = self.relu5(xf)
        xf, mf = self.conv6f(xf, mf)
        xf = self.norm6f(xf)
        xf = self.relu6(xf)
        xf, mf = self.conv7f(xf, mf)

        xb, mb = self.conv1f(xb, mb)
        xb = self.relu1(xb)
        xb, mb = self.conv2f(xb, mb)
        xb = self.norm2f(xb)
        xb = self.relu2(xb)
        xb, mb = self.conv3f(xb, mb)
        xb = self.norm3f(xb)
        xb = self.relu3(xb)
        xb, mb = self.conv4f(xb, mb)
        xb = self.norm4f(xb)
        xb = self.relu4(xb)
        xb, mb = self.conv5f(xb, mb)
        xb = self.norm5f(xb)
        xb = self.relu5(xb)
        xb, mb = self.conv6f(xb, mb)
        xb = self.norm6f(xb)
        xb = self.relu6(xb)
        xb, mb = self.conv7f(xb, mb)

        return x, xf, xb


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=6, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        num_outputs = ndf * min(2 ** n_layers, 8)
        self.D = OrgDiscriminator(input_nc, ndf, n_layers, norm_layer)
        self.convl1 = spectral_norm(nn.Conv2d(num_outputs, num_outputs, kernel_size=1, stride=1))
        self.relul1 = nn.LeakyReLU(0.2)
        self.convl2 = spectral_norm(nn.Conv2d(num_outputs, num_outputs, kernel_size=1, stride=1))
        self.relul2 = nn.LeakyReLU(0.2)
        self.convl3 = nn.Conv2d(num_outputs, 1, kernel_size=1, stride=1)
        self.convg3 = nn.Conv2d(num_outputs, 1, kernel_size=1, stride=1)

    def forward(self, input, mask=None, gp=False, feat_loss=False):

        x, xf, xb = self.D(input, mask)
        feat_l, feat_g = torch.cat([xf, xb]), x
        x = self.convg3(x)

        sim = xf * xb
        sim = self.convl1(sim)
        sim = self.relul1(sim)
        sim = self.convl2(sim)
        sim = self.relul2(sim)
        sim = self.convl3(sim)
        sim_sum = sim
        if not gp:
            if feat_loss:
                return x, sim_sum, feat_g, feat_l
            return x, sim_sum
        return (x + sim_sum) * 0.5
