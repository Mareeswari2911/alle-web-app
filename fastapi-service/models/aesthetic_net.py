import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


IMAGE_SIZE = 224

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])


class NIMA(nn.Module):

    def __init__(self):
        super(NIMA, self).__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features   = mobilenet.features
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x    = self.features(x)
        x    = self.pool(x)
        x    = x.flatten(1)
        dist = self.classifier(x)
        return dist


def preprocess(image):
    if isinstance(image, torch.Tensor):
        pil = transforms.ToPILImage()(image.cpu())
        return _transform(pil).unsqueeze(0)
    elif isinstance(image, np.ndarray):
        pil = Image.fromarray(image.astype(np.uint8))
        return _transform(pil).unsqueeze(0)
    elif isinstance(image, Image.Image):
        return _transform(image).unsqueeze(0)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


class AestheticNet:

    def __init__(self, weights_path=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = NIMA().to(self.device)

        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.K = torch.arange(1, 11, dtype=torch.float32).to(self.device)

    def score(self, image):
        tensor = preprocess(image).to(self.device)
        with torch.no_grad():
            dist = self.model(tensor)
        f_s = (dist * self.K).sum(dim=1)
        return f_s.item()

    def score_distribution(self, image):
        tensor = preprocess(image).to(self.device)
        with torch.no_grad():
            dist = self.model(tensor)
        return dist.squeeze(0).cpu().numpy()

    def reward(self, img_t, img_t1):
        f_t  = self.score(img_t)
        f_t1 = self.score(img_t1)
        r_as = f_t1 - f_t
        return r_as

    def batch_score(self, images):
        images = images.to(self.device)
        with torch.no_grad():
            dist = self.model(images)
        scores = (dist * self.K).sum(dim=1)
        return scores

    def batch_reward(self, imgs_t, imgs_t1):
        f_t  = self.batch_score(imgs_t)
        f_t1 = self.batch_score(imgs_t1)
        return f_t1 - f_t


if __name__ == "__main__":
    net          = AestheticNet()
    dark_image   = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    bright_image = Image.fromarray(np.full((224, 224, 3), 180, dtype=np.uint8))
    score_dark   = net.score(dark_image)
    score_bright = net.score(bright_image)
    print(f"Score dark   : {score_dark:.4f}")
    print(f"Score bright : {score_bright:.4f}")
    r = net.reward(dark_image, bright_image)
    print(f"Reward       : {r:.4f}")
    dist = net.score_distribution(bright_image)
    for k, p in enumerate(dist, start=1):
        print(f"k={k:2d}  P={p:.4f}")