import os
from torch.utils.data import Dataset
from PIL import Image

class ImageDataSet(Dataset):
    def __init__(self, blur_dir, real_dir, transform=None,num_samples=None):
        self.blur_dir = blur_dir
        self.real_dir = real_dir
        self.blur_images = sorted(os.listdir(blur_dir))
        self.real_images = sorted(os.listdir(real_dir))
        self.transform = transform
        self.num_samples = num_samples if num_samples else len(self.blur_images)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        blur_image_path = os.path.join(self.blur_dir, self.blur_images[idx])
        real_image_path = os.path.join(self.real_dir, self.real_images[idx])

        # Görüntüleri yükle
        blur_image = Image.open(blur_image_path).convert('RGB')
        real_image = Image.open(real_image_path).convert('RGB')

        if self.transform:
            blur_image = self.transform(blur_image)
            real_image = self.transform(real_image)

        return blur_image, real_image