from torch.utils.data import DataLoader
from monai.data import CacheDataset
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, SpatialCropd
import os


def get_datalist(source):
    datalist = []
    with open(os.path.join(source, "train.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            files = line.split()
            datalist.append({"image": os.path.join(source, files[0]),
                             "label": os.path.join(source, files[1])})

    print("got datalist, extract: \n", datalist[0])
    return datalist


def get_dataset(source, batch_size=2, limit_sample=None): # Only work for the Task01_BrainTumour dataset for now
    print("loading dataset...")
    datalist = get_datalist(source)
    if limit_sample: datalist = datalist[:limit_sample]

    transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        SpatialCropd(keys=["image", "label"],
                     roi_center=[120, 120, 77],
                     roi_size=[128, 128, 128]) # crop it to make easily usable
    ])

    train_ds = CacheDataset( # CacheDataset might not be a good choice in the future
        data=datalist,
        transform=transform,
        num_workers=8,
        cache_rate=1.0,
    )

    data_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return data_loader
