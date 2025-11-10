from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np

class MySVHN(Dataset):
    def __init__(self, file_path, split="train", download=True, transform=None):
        # split: 'train', 'test', 'extra'
        self.svhn = datasets.SVHN(file_path, split=split, download=download, transform=transform)
        self.targets = np.array(self.svhn.labels)   # SVHN은 .labels에 있음
        self.classes = [str(i) for i in range(10)]  # 0~9 숫자 클래스
        self.ood_class = None  # SVHN은 OOD 클래스 없음

    def __getitem__(self, index):
        data, _ = self.svhn[index]   # transform 적용된 data
        target = self.targets[index]
        if self.ood_class is None:
            return data, target, index
        else:
            return data, target, index, self.ood_class[index]

    def __len__(self):
        return len(self.svhn)

        
