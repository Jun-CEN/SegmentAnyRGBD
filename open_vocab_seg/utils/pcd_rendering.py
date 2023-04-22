import torch
import torch.nn as nn

from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
)


def homogenize_pt(coord):
    return torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)

  
def unproject_pts_pt(intrinsics, coords, depth):
    if coords.shape[-1] == 2:
        coords = homogenize_pt(coords)
    intrinsics = intrinsics.squeeze()[:3, :3]
    coords = torch.inverse(intrinsics).mm(coords.T) * depth.reshape(1, -1)
    return coords.T   # [n, 3]

  
def get_coord_grids_pt(h, w, device, homogeneous=False):
    """
    create pxiel coordinate grid
    :param h: height
    :param w: weight
    :param device: device
    :param homogeneous: if homogeneous coordinate
    :return: coordinates [h, w, 2]
    """
    y = torch.arange(0, h).to(device)
    x = torch.arange(0, w).to(device)
    grid_y, grid_x = torch.meshgrid(y, x)
    if homogeneous:
        return torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1)
    return torch.stack([grid_x, grid_y], dim=-1)  # [h, w, 2]


class PointsRenderer(nn.Module):
    """
    A class for rendering a batch of points. The class should
    be initialized with a rasterizer and compositor class which each have a forward
    function.
    """

    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def to(self, device):
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        r = self.rasterizer.raster_settings.radius

        if type(r) == torch.Tensor:
            if r.shape[-1] > 1:
                idx = fragments.idx.clone()
                idx[idx == -1] = 0
                r = r[:, idx.squeeze().long()]
                r = r.permute(0, 3, 1, 2)

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images


def create_pcd_renderer(h, w, intrinsics, R=None, T=None, radius=None, device="cuda"):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    if R is None:
        R = torch.eye(3)[None]  # (1, 3, 3)
    if T is None:
        T = torch.zeros(1, 3)  # (1, 3)
    cameras = PerspectiveCameras(R=R, T=T,
                                 device=device,
                                 focal_length=((-fx, -fy),),
                                 principal_point=(tuple(intrinsics[:2, -1]),),
                                 image_size=((h, w),),
                                 in_ndc=False,
                                 )

    if radius is None:
        radius = 1.5 / min(h, w) * 2.0

    raster_settings = PointsRasterizationSettings(
        image_size=(h, w),
        radius=radius,
        points_per_pixel=8,
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(1, 1, 1))
    )
    return renderer
