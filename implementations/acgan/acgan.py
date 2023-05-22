# 这段代码是一个基于PyTorch实现的ACGAN (Auxiliary Classifier Generative Adversarial Network) 模型的实现。
# ACGAN是一种生成对抗网络 (GAN) 的变体，它可以生成具有特定属性的图像。
# 这个实现包括了模型的定义和训练过程。具体来说，这个实现包括了以下内容：
# 引入了必要的Python库和PyTorch模块。
# 定义了一个ACGAN模型的生成器和判别器，它们都是由多个卷积层和全连接层组成的神经网络。
# 定义了一个辅助分类器，用于预测生成的图像的属性。
# 定义了损失函数和优化器。
# 加载了MNIST数据集，并将其转换为PyTorch张量。
# 定义了训练过程，包括了生成器和判别器的训练，以及辅助分类器的训练。
# 定义了一些辅助函数，用于保存生成的图像和计算准确率等。
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

# 这行代码使用Python内置的os模块创建一个名为images的文件夹。
# 如果该文件夹已经存在，则不会创建新的文件夹。exist_ok=True参数表示如果文件夹已经存在，则不会引发FileExistsError异常。
# 如果文件夹不存在，则会创建一个新的文件夹。这行代码通常用于在训练神经网络时保存生成的图像或其他数据。
os.makedirs("images", exist_ok=True)

# 这段代码使用Python内置的argparse模块创建了一个命令行参数解析器。
# argparse模块可以帮助我们在命令行中解析参数，并将它们转换为Python对象。
# 在这个例子中，我们定义了一些命令行参数，包括：
# n_epochs：训练的轮数，默认为200。
# batch_size：每个批次的大小，默认为64。
# lr：Adam优化器的学习率，默认为0.0002。
# b1：Adam优化器的一阶动量衰减率，默认为0.5。
# b2：Adam优化器的二阶动量衰减率，默认为0.999。
# n_cpu：用于批量生成的CPU线程数，默认为8。
# latent_dim：潜在空间的维度，默认为100。
# n_classes：数据集的类别数，默认为10。
# img_size：每个图像的大小，默认为32。
# channels：图像的通道数，默认为1。
# sample_interval：每隔多少个批次保存一次生成的图像，默认为400。
# 然后，我们使用parser.parse_args()方法解析命令行参数，并将它们存储在opt对象中。
# 最后，我们打印出opt对象，以便查看解析后的参数。这段代码通常用于在命令行中设置模型的超参数。
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

# 这行代码使用PyTorch的torch.cuda.is_available()方法检查当前系统是否支持CUDA（Compute Unified Device Architecture）。
# 如果系统支持CUDA，则返回True，否则返回False。
# CUDA是NVIDIA开发的一种并行计算平台和编程模型，它可以利用GPU的并行计算能力加速深度学习模型的训练和推理。
# 如果系统支持CUDA，则可以使用PyTorch的CUDA版本来加速模型的训练和推理。
# 在这个例子中，如果系统支持CUDA，则将cuda变量设置为True，否则设置为False。
cuda = True if torch.cuda.is_available() else False

# 这段代码定义了一个函数weights_init_normal，用于初始化神经网络的权重。
# 该函数接受一个神经网络模块m作为输入，并根据模块的类型对其权重进行初始化。
# 具体来说，如果模块是一个卷积层，则使用均值为0、标准差为0.02的正态分布随机初始化其权重；
# 如果模块是一个BatchNorm2d层，则使用均值为1、标准差为0.02的正态分布随机初始化其权重，并将其偏置项初始化为0。
# 这种初始化方法可以帮助神经网络更快地收敛，并且可以避免梯度消失或梯度爆炸的问题。
# 在这个例子中，这个函数通常用于初始化ACGAN模型的生成器和判别器的权重。
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# 这段代码定义了一个ACGAN模型的生成器，它是一个继承自nn.Module的PyTorch模块。
# 生成器的输入是一个噪声向量和一个标签，输出是一个与标签相关的图像。具体来说，生成器包括以下几个部分：
# label_emb：一个nn.Embedding模块，用于将标签转换为嵌入向量。
# l1：一个全连接层，将噪声向量和标签的嵌入向量连接起来，并将其映射到一个大小为128 * self.init_size ** 2的向量。
# conv_blocks：一个由多个卷积层和批归一化层组成的序列，用于将l1的输出转换为一个图像。
# 具体来说，它包括两个上采样层、两个卷积层和一个输出层。
# 其中，第一个卷积层的输入通道数为128，输出通道数为128，卷积核大小为3，步长为1，填充为1；
# 第二个卷积层的输入通道数为64，输出通道数为opt.channels，卷积核大小为3，步长为1，填充为1；
# 输出层使用nn.Tanh()激活函数将输出值限制在[-1, 1]之间。
# 在forward方法中，生成器首先将标签转换为嵌入向量，并将其与噪声向量相乘得到一个生成器输入。
# 然后，生成器将生成器输入传递给全连接层l1，并将其输出重塑为一个大小为[batch_size, 128, init_size, init_size]的张量。
# 最后，生成器将该张量传递给卷积层序列conv_blocks，并返回生成的图像。
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# 这段代码定义了一个ACGAN模型的判别器，它是一个继承自nn.Module的PyTorch模块。判别器的输入是一个图像，输出是一个二元元组，
# 包括一个表示图像真伪的标量和一个表示图像类别的概率向量。具体来说，判别器包括以下几个部分：
# conv_blocks：一个由多个卷积层和批归一化层组成的序列，用于将输入图像转换为一个特征向量。
# 具体来说，它包括四个卷积块，每个卷积块包括一个卷积层、一个LeakyReLU激活函数和一个Dropout2d层。
# 其中，第一个卷积块的输入通道数为opt.channels，输出通道数为16，卷积核大小为3，步长为2，填充为1；
# 其他卷积块的输入通道数和输出通道数分别为上一个卷积块的输出通道数和两倍，卷积核大小为3，步长为2，填充为1。
# 最后，将输出的特征向量展平为一个大小为128 * ds_size ** 2的向量。
# adv_layer：一个全连接层，用于输出一个表示图像真伪的标量。该层的输入大小为128 * ds_size ** 2，输出大小为1，
# 并使用nn.Sigmoid()激活函数将输出值限制在[0, 1]之间。
# aux_layer：一个全连接层，用于输出一个表示图像类别的概率向量。该层的输入大小为128 * ds_size ** 2，输出大小为opt.n_classes，
# 并使用nn.Softmax()激活函数将输出值转换为概率。
# 在forward方法中，判别器首先将输入图像传递给卷积层序列conv_blocks，并将其输出展平为一个特征向量。
# 然后，判别器将该特征向量分别传递给全连接层adv_layer和aux_layer，并返回一个二元元组，
# 包括一个表示图像真伪的标量和一个表示图像类别的概率向量。
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

# 这段代码定义了ACGAN模型的损失函数，包括对抗损失和辅助损失。
# 具体来说，adversarial_loss是一个二元交叉熵损失函数，用于衡量判别器对生成器生成的图像的真伪判断的准确性。
# auxiliary_loss是一个交叉熵损失函数，用于衡量判别器对生成器生成的图像的类别判断的准确性。
# 在训练ACGAN模型时，我们需要同时最小化这两个损失函数。
# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# 这段代码创建了一个ACGAN模型的生成器和判别器实例。
# Generator()和Discriminator()分别是ACGAN模型的生成器和判别器类。
# 在这个例子中，我们创建了一个名为generator的生成器实例和一个名为discriminator的判别器实例，用于训练和测试ACGAN模型。
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()


# 这段代码用于将ACGAN模型的生成器、判别器和损失函数移动到GPU上进行计算。
# 如果系统支持CUDA，则可以使用PyTorch的CUDA版本来加速模型的训练和推理。
# 在这个例子中，如果系统支持CUDA，则将生成器、判别器和损失函数分别移动到GPU上进行计算。
# 具体来说，generator.cuda()、discriminator.cuda()、adversarial_loss.cuda()和auxiliary_loss.cuda()分别将生成器、判别器和两个损失函数移动到GPU上。
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# 这段代码使用apply()方法对ACGAN模型的生成器和判别器的权重进行初始化。
# weights_init_normal是一个自定义的函数，用于初始化神经网络的权重。
# 在这个例子中，weights_init_normal函数使用均值为0、标准差为0.02的正态分布随机初始化卷积层和BatchNorm2d层的权重。
# 这种初始化方法可以帮助神经网络更快地收敛，并且可以避免梯度消失或梯度爆炸的问题。
# apply()方法可以递归地遍历模块树，并对每个模块调用指定的函数。
# 在这个例子中，apply()方法将weights_init_normal函数应用到ACGAN模型的生成器和判别器的所有卷积层和BatchNorm2d层上，以初始化它们的权重。
# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# 这段代码用于配置MNIST数据集的数据加载器。MNIST是一个手写数字识别数据集，包含60000个训练样本和10000个测试样本。
# 在这个例子中，我们使用PyTorch内置的datasets.MNIST类来加载MNIST数据集。
# datasets.MNIST类的参数包括数据集的本地存储路径、是否下载数据集、数据集的预处理方式等。
# 具体来说，我们将MNIST数据集存储在../../data/mnist目录下，设置train=True表示加载训练集，
# 设置download=True表示如果本地没有数据集则自动下载，设置transform参数表示对数据集进行预处理，
# 包括将图像大小调整为opt.img_size、将图像转换为张量、将图像像素值归一化到[-1, 1]之间。
# 最后，我们使用torch.utils.data.DataLoader类来创建一个数据加载器，将MNIST数据集划分为大小为opt.batch_size的小批量数据，并打乱数据集的顺序。
# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# 这段代码定义了ACGAN模型的优化器，包括生成器的优化器optimizer_G和判别器的优化器optimizer_D。
# 在这个例子中，我们使用Adam优化器来优化生成器和判别器的参数。
# Adam是一种自适应学习率优化算法，它可以根据每个参数的梯度自适应地调整学习率。
# torch.optim.Adam是PyTorch中实现Adam优化器的类，它的参数包括要优化的参数、学习率、动量参数beta1和beta2等。
# 在这个例子中，generator.parameters()和discriminator.parameters()分别表示要优化的生成器和判别器的参数，
# opt.lr表示学习率，opt.b1和opt.b2分别表示Adam优化器的动量参数beta1和beta2。
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 这段代码定义了两个PyTorch张量类型FloatTensor和LongTensor，用于在CPU或GPU上存储浮点数和整数数据。
# 具体来说，如果系统支持CUDA，则将FloatTensor和LongTensor分别定义为torch.cuda.FloatTensor和torch.cuda.LongTensor，
# 否则将它们分别定义为torch.FloatTensor和torch.LongTensor。
# 这样做的目的是为了在使用GPU加速计算时，能够将数据存储在GPU上，以提高计算效率。
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# 这段代码定义了一个函数sample_image，用于生成并保存一个n_row x n_row的图像网格。
# 具体来说，该函数的输入参数包括n_row和batches_done，其中n_row表示每行显示的图像数量，batches_done表示已经训练的批次数。
# 在函数内部，首先使用np.random.normal函数生成一个大小为(n_row ** 2, opt.latent_dim)的随机噪声张量z，
# 其中opt.latent_dim表示生成器的输入噪声维度。
# 然后，使用np.array函数生成一个大小为(n_row ** 2,)的标签向量labels，其中每个元素表示对应图像的类别标签，标签范围从0到n_row-1。
# 接下来，将随机噪声张量z和标签向量labels作为输入，调用生成器的generator(z, labels)方法生成n_row ** 2张图像，
# 并将它们保存在images目录下的以batches_done为文件名的png文件中。最后，使用save_image函数将生成的图像网格保存为一个png文件。
def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):  # 遍历所有的训练轮数
    for i, (imgs, labels) in enumerate(dataloader):  # 遍历每个小批量数据

        batch_size = imgs.shape[0]  # 获取当前小批量数据的大小

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)  # 定义一个大小为(batch_size, 1)的张量valid，用于表示真实图像的标签
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)  # 定义一个大小为(batch_size, 1)的张量fake，用于表示生成图像的标签

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))  # 将真实图像转换为FloatTensor类型，并将其封装为一个PyTorch变量real_imgs
        labels = Variable(labels.type(LongTensor))  # 将标签转换为LongTensor类型，并将其封装为一个PyTorch变量labels

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()  # 清空生成器的梯度缓存

        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))  # 从标准正态分布中采样大小为(batch_size, opt.latent_dim)的随机噪声张量z
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))  # 从0到opt.n_classes-1中随机采样大小为batch_size的标签向量gen_labels

        gen_imgs = generator(z, gen_labels)  # 生成器生成一批图像gen_imgs

        validity, pred_label = discriminator(gen_imgs)  # 判别器对生成的图像进行判别，得到判别结果validity和预测标签pred_label

        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))  # 计算生成器的损失函数g_loss

        g_loss.backward()  # 反向传播计算梯度
        optimizer_G.step()  # 更新生成器的参数

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()  # 清空判别器的梯度缓存

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)  # 判别器对真实图像进行判别，得到判别结果real_pred和预测标签real_aux
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2  # 计算真实图像的判别器损失函数d_real_loss

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())  # 判别器对生成图像进行判别，得到判别结果fake_pred和预测标签fake_aux
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2  # 计算生成图像的判别器损失函数d_fake_loss

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2  # 计算判别器的总损失函数d_loss

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)  # 将真实图像和生成图像的预测标签拼接成一个数组pred
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)  # 将真实图像和生成图像的标签拼接成一个数组gt
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)  # 计算判别器的准确率d_acc

        d_loss.backward()  # 反向传播计算梯度
        optimizer_D.step()  # 更新判别器的参数

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
        )  # 打印当前训练轮数、批次数、判别器损失函数、判别器准确率和生成器损失函数

        batches_done = epoch * len(dataloader) + i  # 计算已经训练的批次数
        if batches_done % opt.sample_interval == 0:  # 如果已经训练的批次数是opt.sample_interval的倍数
            sample_image(n_row=10, batches_done=batches_done)  # 生成并保存一张10x10的图像网格
