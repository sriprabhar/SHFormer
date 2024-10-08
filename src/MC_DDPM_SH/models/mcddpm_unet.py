from models.guided_ddpm_unet import UNetModel
from utils.mri_data_utils.transform_util import *
import torch.nn.functional as F

class KspaceModel(UNetModel):
    """
    A UNetModel that performs on kspace data. Expects extra kwargs `kspace_zf`, `image_zf`, `mask_c`.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        assert in_channels == 2, "mri image is considered"
        # we use in_channels * 2 because image_zf is also input.
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, kspace_zf, image_zf, mask_c):
        """

        :param x: the [N x 2 x H x W] tensor of inputs, x_t at time t.
        :param timesteps: a batch of timestep indices.
        :param kspace_zf: the [N x 2 x H x W] tensor, zero-filling kspace data.
        :param image_zf: the [N x 2 x H x W] tensor, zero-filling reconstruction.
        :param mask_c: the [N x 2 x H x W] tensor with value of 0 or 1, equals to 1 - mask.
        :return: noise estimation or score function estimation in unsampled position of kspace data.
        """
        x_full = x + kspace_zf
        # permute operation is for fft operation.
        image_full = ifftc_th(x_full)
        x = th.cat([image_full, image_zf], dim=1)
#         print(f'kspacemodel, x: {x.dtype}, {x.shape}')
        x = F.pad(x,(5,5,5,5),"constant",0) # for cardiac (150,150) --> (160,160)
        output = super().forward(x, timesteps)
        output = output[:,:,5:-5,5:-5] # for cardiac (160,160) --> (150,150)
        return fftc_th(output) * mask_c
