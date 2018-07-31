import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def open_cv_test():
    img_working = np.array(
        [[[0., 0., 0.],
          [0., 0., 0.],
          [0.00766595, 0.00766595, 0.00766595],
          [0.39684995, 0.39684995, 0.39684995],
          [0.02028469, 0.02028469, 0.02028469]],
         [[0., 0., 0.],
          [0.02803903, 0.02803903, 0.02803903],
          [0.04867653, 0.04867653, 0.04867653],
          [0.00963292, 0.00963292, 0.00963292],
          [0., 0., 0.]],
         [[0., 0., 0.],
          [0.0592962, 0.0592962, 0.0592962],
          [0.41714191, 0.41714191, 0.41714191],
          [0., 0., 0.],
          [0., 0., 0.]],
         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],
         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]])

    # print(img_working.shape)
    # plt.imshow(img_working)
    # plt.show()

    img = np.array([[[[0.0000, 0.0000, 0.0077, 0.3968, 0.0203],
                     [0.0000, 0.0280, 0.0487, 0.0096, 0.0000],
                     [0.0000, 0.0593, 0.4171, 0.0000, 0.0000],
                     [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                     [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]])
    print(img.shape)
    img = torch.from_numpy(img)
    # Need this to show with matplot lib (it makes 3 channels)
    # img = torch.cat((img, img, img), 1)
    img = img[0].numpy()
    img = np.transpose(img, (1, 2, 0))

    # print(img.shape)
    # plt.imshow(img)
    # plt.show()

    import cv2
    print(img)
    img = img*255
    print(img)
    img = np.around(img).astype(np.uint8)
    print(img)
    print(img.shape)
    equ = cv2.equalizeHist(img)
    print(equ.shape)
    equ = equ.reshape((1, 1, 5, 5))
    print(equ.shape)

    equ = torch.from_numpy(equ)
    equ = torch.cat((equ, equ, equ), 1)
    equ = equ[0].numpy()
    equ = np.transpose(equ, (1, 2, 0))
    cv2.imwrite('equ.png', equ)

    print(equ.shape)
    plt.imshow(equ)
    plt.show()

    # print(img.shape)


def test_cat():
    # tensor of 2x1x3x3 dimension
    x = torch.tensor([[[[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]]],
                      [[[9, 10, 11],
                        [12, 13, 14],
                        [15, 16, 17]]]])
    y = torch.tensor([[[[100, 101, 102],
                        [103, 104, 105],
                        [106, 107, 108]]],
                      [[[109, 110, 111],
                        [112, 113, 114],
                        [115, 116, 117]]]])
    print(x.shape, y.shape)
    print(x)
    print(y)

    xy = torch.cat((x, y), 1)
    print(xy)
    print(xy.shape)

    grid = torchvision.utils.make_grid(x, padding=0, nrow=2)
    print(grid.shape)


if __name__ == '__main__':
    # test_cat()
    open_cv_test()
