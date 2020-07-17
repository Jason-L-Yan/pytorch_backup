from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
import cv2


class RAMDataset(Dataset):
    """将数据读取到内存(RAM)中，防止 SSD(ROM)限制"""
    def __init__(self, image_fnames, targets):
        super(RAMDataset, self).__init__()
        self.targets = targets
        self.images = []
        for fname in tqdm(image_fnames, desc="Loading files in RAM"):  # desc:传入str类型，作为进度条标题（类似于说明）
            with open(fname, "rb") as f:
                self.images.append(f.read())

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        targets = self.targets[index]
        image, retval = cv2.imdecode(self.images[index], cv2.IMREAD_COLOR)
        return image, targets