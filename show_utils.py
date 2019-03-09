import matplotlib.pyplot as plt
import itertools
import torch
from torch.autograd import Variable

fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)    # fixed noise
with torch.no_grad():
  fixed_z_ = Variable(fixed_z_.cuda())


def show_result(num_epoch, G, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    with torch.no_grad():
        z_ = Variable(z_.cuda())

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)  # shape = (25,1,64,64)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)  # x 轴不可见
        ax[i, j].get_yaxis().set_visible(False)  # y 轴不可见

    for k in range(5*5):  # 显示出25张图片
        i = k // 5
        j = k % 5
        ax[i, j].cla()  # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()