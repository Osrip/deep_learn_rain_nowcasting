import torch

from modules_blocks import Network


def main():

    # device = torch.device('cuda')
    device = torch.device('cpu')

    # test_tensor = torch.randn((3, 3, 32, 32), device=device)
    test_tensor = torch.randn((4, 8, 256, 256), device=device)

    model = Network(c_in=8, s_width_height_in=256)
    x = model(test_tensor)
    pass


if __name__ == '__main__':
    main()


