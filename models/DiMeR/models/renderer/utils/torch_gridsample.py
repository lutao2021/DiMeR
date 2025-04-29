import torch

def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    # ix = ((ix + 1) / 2) * (IW - 1)
    # iy = ((iy + 1) / 2) * (IH - 1)

    # align_corners=False
    ix = 0.5 * ((ix + 1.0) * IW - 1.0)
    iy = 0.5 * ((iy + 1.0) * IH - 1.0)

    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    # Apply zero-padding by clamping indices within image bounds
    with torch.no_grad():

        # Create a mask to set out-of-bounds areas to zero (zero-padding)
        mask_nw = (ix_nw >= 0) & (ix_nw < IW) & (iy_nw >= 0) & (iy_nw < IH)
        mask_ne = (ix_ne >= 0) & (ix_ne < IW) & (iy_ne >= 0) & (iy_ne < IH)
        mask_sw = (ix_sw >= 0) & (ix_sw < IW) & (iy_sw >= 0) & (iy_sw < IH)
        mask_se = (ix_se >= 0) & (ix_se < IW) & (iy_se >= 0) & (iy_se < IH)

        ix_nw = torch.clamp(ix_nw, 0, IW - 1)
        iy_nw = torch.clamp(iy_nw, 0, IH - 1)
        ix_ne = torch.clamp(ix_ne, 0, IW - 1)
        iy_ne = torch.clamp(iy_ne, 0, IH - 1)
        ix_sw = torch.clamp(ix_sw, 0, IW - 1)
        iy_sw = torch.clamp(iy_sw, 0, IH - 1)
        ix_se = torch.clamp(ix_se, 0, IW - 1)
        iy_se = torch.clamp(iy_se, 0, IH - 1)

    # Convert image to 1D for easier indexing
    image = image.view(N, C, IH * IW)

    # Gather the values using the clamped indices, ensuring that out-of-bound indices are set to 0
    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    nw_val *= mask_nw.float()
    ne_val *= mask_ne.float()
    sw_val *= mask_sw.float()
    se_val *= mask_se.float()

    # Compute the output value
    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

