from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, SpatialCropd, ConcatItemsd
import os


def get_datalist(source):
    datalist = []
    with open(os.path.join(source, "train.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            file_dic = {}
            files = line.split()
            for i, file in enumerate(files[:-1]):
                file_dic[f"channel_{i}"] = os.path.join(source, file)

            file_dic["label"] = os.path.join(source, files[-1])
            datalist.append(file_dic)

    print("got datalist, extract: \n", datalist[0])
    return datalist


def get_dataset(source, batch_size=2, limit_sample=None): # Only work for the Task01_BrainTumour dataset for now
    print("loading dataset...")
    datalist = get_datalist(source)
    keys = list(datalist[0].keys())
    if limit_sample: datalist = datalist[:limit_sample]

    transform = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ConcatItemsd(keys[:-1], "input"),
        SpatialCropd(keys=['input', 'label'], # crop it to make easily usable for etape 1
                     roi_size=[128, 128, 128],
                     roi_center=[0, 0, 0]
                     )
    ])

    train_ds = Dataset( # A more optimized dataset can be used
        data=datalist,
        transform=transform
    )

    data_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return data_loader
