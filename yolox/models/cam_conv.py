from torch import nn
import torch

# Some helper functions
# ed = expand last dim
ed = lambda x: torch.unsqueeze(x, -1)
ed2 = lambda x: ed(ed(x))
ed3 = lambda x: ed(ed2(x))


def convert_NCHW_to_NHWC(inp):
    """Convert the tensor from caffe format NCHW into tensorflow format NHWC

        inp: tensor
    """
    return torch.transpose(inp, [0, 2, 3, 1])


def convert_NHWC_to_NCHW(inp):
    """Convert the tensor from tensorflow format NHWC into caffe format NCHW

        inp: tensor
    """
    return torch.transpose(inp, [0, 3, 1, 2])


class CamConv(nn.Module):
    """ Add Camera Coord Maps to a tensor
    """

    def __init__(self, ):
        # resize_policy=tf.image.ResizeMethod.BILINEAR):
        super().__init__()

    def _resize_map_(self, data, w, h):
        if self.data_format == 'channels_first':
            data_cl = convert_NCHW_to_NHWC(data)  # data to channels last
            data_cl_r = nn.functional.interpolate(data_cl, (h, w), mode_function='bilinear',
                                                  align_corners=True)  # tf.image.resize_images(data, [h, w], method=self.resize_policy, align_corners=True)
            return convert_NHWC_to_NCHW(data_cl_r)
        else:
            # return tf.image.resize_images(data, [h, w], method=self.resize_policy, align_corners=True)
            return nn.functional.interpolate(data, (h, w), mode_function='bilinear',
                                                  align_corners=True)

    def __define_coord_channels__(self, n, x_dim, y_dim):
        """
        Returns coord x and y channels from 0 to x_dim-1 and from 0 to y_dim -1
        """
        xx_ones = torch.ones([n, y_dim], dtype=torch.int32)
        xx_ones = torch.unsqueeze(xx_ones, -1)

        x_range = torch.range(x_dim)
        x_range = torch.unsqueeze(x_range, 0)

        xx_range = torch.tile(x_range, [n, 1])
        xx_range = torch.unsqueeze(xx_range, 1)
        xx_channel = torch.matmul(xx_ones, xx_range)

        yy_ones = torch.ones([n, x_dim], dtype=torch.int32)
        yy_ones = torch.unsqueeze(yy_ones, 1)
        yy_range = torch.tile(torch.unsqueeze(torch.range(y_dim), 0), [n, 1])
        yy_range = torch.unsqueeze(yy_range, -1)
        yy_channel = torch.matmul(yy_range, yy_ones)

        if self.data_format == 'channels_last':
            xx_channel = torch.unsqueeze(xx_channel, -1)
            yy_channel = torch.unsqueeze(yy_channel, -1)
        else:
            xx_channel = torch.unsqueeze(xx_channel, 1)
            yy_channel = torch.unsqueeze(yy_channel, 1)

        xx_channel = torch.cast(xx_channel, 'float32')
        yy_channel = torch.cast(yy_channel, 'float32')
        return xx_channel, yy_channel

    # def call(self, input_tensor,h,w,cx,cy,fx,fy):
    def forward(self, input_tensor, h=0, w=0, cx=0, cy=0, fx=0, fy=0):
        """
        input_tensor: Tensor
            (N,H,W,C) if channels_last or (N,C,H,W) if channels_first
        """
        print(input_tensor.shape)
        batch_size_tensor = input_tensor.shape[0]  # tf.shape(input_tensor)[0]
        x_dim_tensor = input_tensor.shape[2]  # tf.shape(input_tensor)[2]
        y_dim_tensor = input_tensor.shape[1]  # tf.shape(input_tensor)[1]
        ax_concat = -1
        xx_channel, yy_channel = self.__define_coord_channels__(batch_size_tensor, w, h)

        extra_channels = []
        # 1) Normalized coordinates
        if self.norm_coord_maps:
            norm_xx_channel = (xx_channel / (w - 1)) * 2.0 - 1.0
            norm_yy_channel = (yy_channel / (h - 1)) * 2.0 - 1.0
            if self.with_r:
                norm_rr_channel = torch.sqrt(torch.square(norm_xx_channel - 0.5) + torch.square(norm_yy_channel - 0.5))
                extra_channels = extra_channels + [norm_xx_channel, norm_yy_channel, norm_rr_channel]
            else:
                extra_channels = extra_channels + [norm_xx_channel, norm_yy_channel]

        if self.centered_coord or self.fov_maps:
            # 2) Calculate Centered Coord
            # ed2 is equal to extend_dims twice
            cent_xx_channel = (xx_channel - ed2(cx) + 0.5)
            cent_yy_channel = (yy_channel - ed2(cy) + 0.5)

            # 3) Field of View  coordinates
            if self.fov_maps:
                fov_xx_channel = torch.atan(cent_xx_channel / ed2(fx))
                fov_yy_channel = torch.atan(cent_yy_channel / ed2(fy))
                extra_channels = extra_channels + [fov_xx_channel, fov_yy_channel]
            # 4) Scaled Centered  coordinates
            if self.centered_coord:
                extra_channels = extra_channels + [cent_xx_channel / self.scale_centered_coord,
                                                   cent_yy_channel / self.scale_centered_coord]

        # 5) Coord Maps (Unormalized, uncentered and unscaled)
        if self.coord_maps:
            extra_channels = extra_channels + [xx_channel, yy_channel]

        # Concat and resize
        if len(extra_channels) > 0:
            extra_channels = torch.concat(extra_channels, axis=ax_concat)
            extra_channels = self._resize_map_(extra_channels, x_dim_tensor, y_dim_tensor)
            extra_channels = [extra_channels]
        # 6) Distance to border in pixels in feature space.
        if self.bord_dist:
            t_xx_channel, t_yy_channel = self.__define_coord_channels__(batch_size_tensor, x_dim_tensor, y_dim_tensor)
            l_dist = t_xx_channel
            r_dist = torch.cast(x_dim_tensor, torch.float32) - t_xx_channel - 1
            t_dist = t_yy_channel
            b_dist = torch.cast(y_dim_tensor, torch.float32) - t_yy_channel - 1
            extra_channels = extra_channels + [l_dist, r_dist, t_dist, b_dist]

        extra_channels = [torch.stop_gradient(e) for e in extra_channels]  # Stop Gradients
        output_tensor = torch.concat(extra_channels + [input_tensor], axis=ax_concat)
        return output_tensor

