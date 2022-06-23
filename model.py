from PIL import Image
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models
from tqdm import tqdm


class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.vgg = models.vgg19(pretrained=True).features
        # for param in self.vgg.parameters():
        #     param.requires_grad_(False)
        self.vgg = torch.load('model.h5')
        self.style_weights = {'conv1_1': 1., 'conv2_1': 0.8, 'conv3_1': 0.5, 'conv4_1': 0.3, 'conv5_1': 0.1}
        self.content_weight = 1
        self.style_weight = 1e6

    def load_image(self, img_path, max_size=400, shape=None):
        '''
            Load in and transform an image, making sure the image
            is <= 400 pixels in the x-y dims.
        '''
        image = Image.open(img_path).convert('RGB')
        size = shape if shape else min(max_size, max(image.size))
        in_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        # discard the transparent, alpha channel (that's the :3) and add the batch dimension
        image = in_transform(image)[:3, :, :].unsqueeze(0)
        return image


    def get_features(self, image, layers=None):
        """
            Run an image forward through a model and get the features for
            a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
        """
        # if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if str(name) in layers:
                features[layers[name]] = x

        return features

    def gram_matrix(self, m):
        """
            Calculate the Gram Matrix of a given tensor
            Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
        """
        b, d, h, w = m.size()
        m = m.view(d, h * w)
        gram = torch.mm(m, m.t())
        return gram

    def im_convert(self, m):
        """
            Display a tensor as an image.
        """
        image = m.to("cpu").clone().detach().numpy().squeeze().transpose(1, 2, 0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        return image.clip(0, 1)

    def stylize(self, steps):
        """
            Train model to stylize.
        """
        content = self.load_image('content.jpg').to(self.device)
        style = self.load_image('style.jpg', shape=content.shape[-2:]).to(self.device)
        target = content.clone().requires_grad_(True).to(self.device)
        optimizer = optim.Adam([target], lr=0.13)
        content_features = self.get_features(content, self.vgg)
        style_features = self.get_features(style, self.vgg)
        style_grams = {layer: self.gram_matrix(style_features[layer]) for layer in style_features}
        for _ in tqdm(range(1, steps + 1)):
            target_features = self.get_features(target, self.vgg)
            content_loss = torch.mean((target_features["conv4_2"] - content_features["conv4_2"]) ** 2)
            style_loss = 0
            for layer in self.style_weights:
                target_feature = target_features[layer]
                _, d, h, w = target_feature.shape
                target_gram = self.gram_matrix(target_feature)
                style_gram = style_grams[layer]
                layer_style_loss = self.style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
                style_loss += layer_style_loss / (d * h * w)
            total_loss = self.content_weight * content_loss + self.style_weight * style_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            img = Image.fromarray((self.im_convert(target) * 255).astype(np.uint8))
            img.save('res.jpg')
            return img