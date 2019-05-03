class Config:
    __bn           = True
    __dp           = 0.2
    __n_layers     = 3
    __filters_base = 32
    __patch_border = 16
    __pad          = 1
    __n_classes    = 2
    __n_channels   = 3
    __n_epoch      = 8

    def __init__(self, ID, bn=__bn, dp=__dp, n_layers=__n_layers, filters_base=__filters_base, n_epoch=__n_epoch,
                 patch_border=__patch_border, pad=__pad, n_classes=__n_classes, n_channels=__n_channels):

        self.ID           = ID
        self.bn           = bn
        self.dp           = dp
        self.n_layers     = n_layers
        self.n_epoch      = n_epoch
        self.filters_base = filters_base
        self.patch_border = patch_border
        self.pad          = pad
        self.n_classes    = n_classes
        self.n_channels   = n_channels


class Metadata:
    def __init__(self, k, n_rot, n_crop, crop_sz, step_sz, img_sz):
        self.k     = k
        self.n_rot = n_rot
        self.n_crop = n_crop
        self.crop_sz = crop_sz
        self.step_sz = step_sz
        self.img_sz = img_sz
