import os, time, sys
import pickle
import imageio
import torch
import torch.optim as optim
from torch.autograd import Variable
from model import generator, discriminator, BCE_loss
from show_utils import show_result, show_train_hist
import mydata_loader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 20
img_size = 64
# data_loader
celea_data = '/home/zzp/SSD_ping/my-root-path/My-core-python/DATA/CelebA/Img/img_align_celeba/'


def train(dataset='mnist'):
    if dataset == 'mnist':
        mnist_loader = mydata_loader.mnist_loader(img_size, batch_size)
        loader = mnist_loader
    elif dataset == 'celeba':
        data_dir = celea_data
        celeba_loader = mydata_loader.celeba_loader(img_size, batch_size, data_dir)
        loader = celeba_loader
    else:
        sys.stderr.write('Error! the dataset name must be mnist or celeba! please input the right name !!!')
        sys.exit(1)

    # network
    G = generator(128, dataset)
    D = discriminator(128, dataset)
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.cuda()
    D.cuda()

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # results save folder
    if not os.path.isdir(f'{dataset}_DCGAN_results'):
        os.mkdir(f'{dataset}_DCGAN_results')
    if not os.path.isdir(f'{dataset}_DCGAN_results/Random_results'):
        os.mkdir(f'{dataset}_DCGAN_results/Random_results')
    if not os.path.isdir(f'{dataset}_DCGAN_results/Fixed_results'):
        os.mkdir(f'{dataset}_DCGAN_results/Fixed_results')

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    # ****************************** start training ! *********************************************************
    print('training start!')
    start_time = time.time()
    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []
        epoch_start_time = time.time()
        for x_data, _ in loader:

            # ************************************* train discriminator D *************************
            D.zero_grad()

            mini_batch = x_data.size()[0]

            y_real = torch.ones(mini_batch)
            y_fake = torch.zeros(mini_batch)

            x_data, y_real, y_fake = Variable(x_data.cuda()), Variable(y_real.cuda()), Variable(y_fake.cuda())
            D_result = D(x_data).squeeze()
            D_real_loss = BCE_loss(D_result, y_real)

            z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z = Variable(z.cuda())
            G_result = G(z)
            D_result = D(G_result).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake)

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.data.item())

            # ****************************** train generator G *****************************
            G.zero_grad()

            z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z = Variable(z.cuda())

            G_result = G(z)
            D_result = D(G_result).squeeze()
            G_train_loss = BCE_loss(D_result, y_real)
            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.data.item())

        epoch_end_time = time.time()
        per_epoch_time = epoch_end_time - epoch_start_time

        print(f'[{epoch + 1}/{train_epoch}] - per-epoch-time: {per_epoch_time:.2f},\t '
              f'loss_d: {torch.mean(torch.FloatTensor(D_losses)):.3f}, \t'
              f'loss_g: {torch.mean(torch.FloatTensor(G_losses)):.3f}')

        p = f'{dataset}_DCGAN_results/Random_results/MNIST_DCGAN_' + str(epoch + 1) + '.png'
        fixed_p = f'{dataset}_DCGAN_results/Fixed_results/MNIST_DCGAN_' + str(epoch + 1) + '.png'
        show_result((epoch+1), G, save=True, path=p, isFix=False)
        show_result((epoch+1), G, save=True, path=fixed_p, isFix=True)

        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_time)

    end_time = time.time()
    total_time = end_time - start_time
    train_hist['total_time'].append(total_time)

    print(f"Avg per epoch ptime: {torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])):.2f},"
          f" total {train_epoch} epochs ptime: {total_time:.2f}")

    print("Training finish!... save training results")

    # save parameters
    torch.save(G.state_dict(), f'{dataset}_DCGAN_results/generator_param.pkl')
    torch.save(D.state_dict(), f'{dataset}_DCGAN_results/discriminator_param.pkl')
    with open(f'{dataset}_DCGAN_results/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path=f'{dataset}_DCGAN_results/MNIST_DCGAN_train_hist.png')

    images1 = []
    images2 = []
    for e in range(train_epoch):
        img_name1 = f'{dataset}_DCGAN_results/Fixed_results/MNIST_DCGAN_' + str(e + 1) + '.png'
        img_name2 = f'{dataset}_DCGAN_results/Random_results/MNIST_DCGAN_' + str(e + 1) + '.png'
        images1.append(imageio.imread(img_name1))
        images1.append(imageio.imread(img_name2))
    imageio.mimsave(f'{dataset}_DCGAN_results/generation_animation_fixed.gif', images1, fps=5)
    imageio.mimsave(f'{dataset}_DCGAN_results/generation_animation_random.gif', images2, fps=5)


if __name__ == '__main__':
    train(dataset='celeba')
