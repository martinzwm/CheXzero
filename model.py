"""
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from bidirectional_cross_attention import BidirectionalCrossAttention

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=False)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=False)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
       # print(self.positional_embedding.shape,x.shape)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.video_mask = torch.ones((1, 4096)).bool()
        # self.text_mask = torch.ones((1, 8192)).bool()

        # self.joint_cross_attn = BidirectionalCrossAttention(
        #     dim = 512,
        #     heads = 8,
        #     dim_head = 64,
        #     context_dim = 386
        # )


        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, batch):
        image, text,zeroshot_weights = batch
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
      #  image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        # print(image_features.shape)
        # print(text_features.shape)
        # shape = [global_batch_size, global_batch_size]
     #   norms = torch.linalg.norm(image_features,dim=-1, keepdim=True)
        image_features = image_features # (1, 768)

        # obtain logits
        logits = image_features @ zeroshot_weights # (1, num_classes)
        logits = torch.squeeze(logits, axis=0) # (num_classes,)
      #  print(logits.shape,'logitsshape')
        return logits


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
import subprocess
import numpy as np
import os
import sys
sys.path.append('/opt/conda/envs/biovil/lib/python3.9/site-packages') # add path to biovil text encoder

import pandas as pd
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from typing import List, Tuple

import torch
from torch.utils import data
from tqdm.notebook import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score

import clip
from model import CLIP
from eval import evaluate, plot_roc, accuracy, sigmoid, bootstrap, compute_cis

from health_multimodal.text.utils import get_cxr_bert

CXR_FILEPATH = '../../project-files/data/test_cxr.h5'
FINAL_LABEL_PATH = '../../project-files/data/final_paths.csv'

class CXRTestDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        img_path: Path to hdf5 file containing images.
        label_path: Path to file containing labels 
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(
        self, 
        img_path: str, 
        transform = None, 
    ):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']
        self.transform = transform
            
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx] # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img) # torch, (320, 320)
        
        if self.transform:
            img = self.transform(img)
            
        sample = {'img': img}
    
        return sample

def load_clip(model_path, pretrained=False, context_length=77, change_text_encoder=False): 
    """
    FUNCTION: load_clip
    ---------------------------------
    """
    device = torch.device("cuda:0")
    if pretrained is False: 
        # use new model params
        params = {
            'embed_dim':768,
            'image_resolution': 320,
            'vision_layers': 12,
            'vision_width': 768,
            'vision_patch_size': 16,
            'context_length': context_length, 
            'vocab_size': 49408,
            'transformer_width': 512,
            'transformer_heads': 8,
            'transformer_layers': 12
        }

        model = CLIP(**params)
    else: 
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False) 
        if change_text_encoder:
            tokenizer, text_model = get_cxr_bert()
            model.encode_text = text_model
        try: 
            print(model_path,"using an online model  ")
            # model=torch.load(model_path).to(device)
            model.load_state_dict(torch.load(model_path))
            model = model.to(device)
        except: 
            print("Argument error. Set pretrained = True.", sys.exc_info()[0])
            raise
    return model

def zeroshot_classifier(classnames, templates, model, context_length=77):
    """
    FUNCTION: zeroshot_classifier
    -------------------------------------
    This function outputs the weights for each of the classes based on the 
    output of the trained clip model text transformer. 
    
    args: 
    * classnames - Python list of classes for a specific zero-shot task. (i.e. ['Atelectasis',...]).
    * templates - Python list of phrases that will be indpendently tested as input to the clip model.
    * model - Pytorch model, full trained clip model.
    * context_length (optional) - int, max number of tokens of text inputted into the model.
    
    Returns PyTorch Tensor, output of the text encoder given templates. 
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        zeroshot_weights = []
        # compute embedding through model for each class
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] # format with class
            texts = clip.tokenize(texts, context_length=context_length) # tokenize
            texts = texts.to(device)
            class_embeddings = model.encode_text(texts) # embed with text encoder
            
            # normalize class_embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # average over templates 
            class_embedding = class_embeddings.mean(dim=0) 
            # norm over new averaged templates
            class_embedding /= class_embedding.norm() 
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights

def predict(loader, model, zeroshot_weights, softmax_eval=True, verbose=0): 
    """
    FUNCTION: predict
    ---------------------------------
    This function runs the cxr images through the model 
    and computes the cosine similarities between the images
    and the text embeddings. 
    
    args: 
        * loader -  PyTorch data loader, loads in cxr images
        * model - PyTorch model, trained clip model 
        * zeroshot_weights - PyTorch Tensor, outputs of text encoder for labels
        * softmax_eval (optional) - Use +/- softmax method for evaluation 
        * verbose (optional) - bool, If True, will print out intermediate tensor values for debugging.
        
    Returns numpy array, predictions on all test data samples. 
    """
    device = next(model.parameters()).device
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            images = data['img'].to(device)

            # predict
            image_features = model.encode_image(images) 
            image_features /= image_features.norm(dim=-1, keepdim=True) # (1, 768)

            # obtain logits
            logits = image_features @ zeroshot_weights # (1, num_classes)
            logits = np.squeeze(logits.cpu().numpy(), axis=0) # (num_classes,)
        
            if softmax_eval is False: 
                norm_logits = (logits - logits.mean()) / (logits.std())
                logits = sigmoid(norm_logits) 
            
            y_pred.append(logits)
            
            if verbose: 
                plt.imshow(images[0][0])
                plt.show()
                print('images: ', images)
                print('images size: ', images.size())
                
                print('image_features size: ', image_features.size())
                print('logits: ', logits)
                print('logits size: ', logits.size())
         
    y_pred = np.array(y_pred)
    return np.array(y_pred)

def run_single_prediction(cxr_labels, template, model, loader, softmax_eval=True, context_length=77): 
    """
    FUNCTION: run_single_prediction
    --------------------------------------
    This function will make probability predictions for a single template
    (i.e. "has {}"). 
    
    args: 
        * cxr_labels - list, labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * template - string, template to input into model. 
        * model - PyTorch model, trained clip model
        * loader - PyTorch data loader, loads in cxr images
        * softmax_eval (optional) - Use +/- softmax method for evaluation 
        * context_length (optional) - int, max number of tokens of text inputted into the model.
        
    Returns list, predictions from the given template. 
    """
    cxr_phrase = [template]
    # zeroshot_weights [=] (dim of encoded text, num_classes), for chexzero, dim of encoded text = 512
    zeroshot_weights = zeroshot_classifier(cxr_labels, cxr_phrase, model, context_length=context_length)
    # y_pred [=] (num_samples, num_classes)
    y_pred = predict(loader, model, zeroshot_weights, softmax_eval=softmax_eval)
    return y_pred

def process_alt_labels(alt_labels_dict, cxr_labels): 
    """
        Process alt labels and return relevant info. If `alt_labels_dict` is 
        None, return None. 
    
    Returns: 
    * alt_label_list : list
        List of all alternative labels
    * alt_label_idx_map : dict
        Maps alt label to idx of original label in cxr_labels
        Needed to access correct column during evaluation
       
    """
    
    if alt_labels_dict is None: 
        return None, None
    
    def get_inverse_labels(labels_alt_map: dict): 
        """
        Returns dict mapping alternative label back to actual label. 
        Used for reference during evaluation.
        """
        inverse_labels_dict  = {}
        for main in labels_alt_map:
            inverse_labels_dict[main] = main # adds self to list of alt labels
            for alt in labels_alt_map[main]:
                inverse_labels_dict[alt] = main
        return inverse_labels_dict
    
    inv_labels_dict = get_inverse_labels(alt_labels_dict)
    alt_label_list = [w for w in inv_labels_dict.keys()]
    
    # create index map
    index_map = dict()
    for i, label in enumerate(cxr_labels): 
          index_map[label] = i
    
    # make map to go from alt label directly to index
    alt_label_idx_map = dict()
    for alt_label in alt_label_list: 
        alt_label_idx_map[alt_label] = index_map[inv_labels_dict[alt_label]]
    
    return alt_label_list, alt_label_idx_map 

def run_softmax_eval(model, loader, eval_labels: list, pair_template: tuple, context_length: int = 77): 
    """
    Run softmax evaluation to obtain a single prediction from the model.
    """
     # get pos and neg phrases
    pos = pair_template[0]
    neg = pair_template[1]

    # get pos and neg predictions, (num_samples, num_classes)
    pos_pred = run_single_prediction(eval_labels, pos, model, loader, 
                                     softmax_eval=True, context_length=context_length) 
    neg_pred = run_single_prediction(eval_labels, neg, model, loader, 
                                     softmax_eval=True, context_length=context_length) 

    # compute probabilities with softmax
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    y_pred = np.exp(pos_pred) / sum_pred
    return y_pred
    
def run_experiment(model, cxr_labels, cxr_templates, loader, y_true, alt_labels_dict=None, softmax_eval=True, context_length=77, use_bootstrap=True): 
    '''
    FUNCTION: run_experiment
    ----------------------------------------
    This function runs the zeroshot experiment on each of the templates
    individually, and stores the results in a list. 
    
    args: 
        * model - PyTorch model, trained clip model 
        * cxr_labels - list, labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * cxr_templates - list, templates to input into model. If softmax_eval is True, 
        this should be a list of tuples, where each tuple is a +/- pair
        * loader - PyTorch data loader, loads in cxr images
        * y_true - list, ground truth labels for test dataset
        * softmax_eval (optional) - bool, if True, will evaluate results through softmax of pos vs. neg samples. 
        * context_length - int, max number of tokens of text inputted into the model. 
        * use_bootstrap (optional) - bool, whether or not to use bootstrap sampling
        
    Returns a list of results from the experiment. 
    '''
    
    alt_label_list, alt_label_idx_map = process_alt_labels(alt_labels_dict, cxr_labels)
    if alt_label_list is not None: 
        eval_labels = alt_label_list
    else: 
        eval_labels = cxr_labels 
    
    results = []
    for template in cxr_templates:
        print('Phrase being used: ', template)
        
        try: 
            if softmax_eval: 
                y_pred = run_softmax_eval(model, loader, eval_labels, template, context_length=context_length)
                
            else: 
                # get single prediction
                y_pred = run_single_prediction(eval_labels, template, model, loader, 
                                               softmax_eval=softmax_eval, context_length=context_length)
#             print("y_pred: ", y_pred)
        except: 
            print("Argument error. Make sure cxr_templates is proper format.", sys.exc_info()[0])
            raise
    
        # evaluate
        if use_bootstrap: 
            # compute bootstrap stats
            boot_stats = bootstrap(y_pred, y_true, eval_labels, label_idx_map=alt_label_idx_map)
            results.append(boot_stats) # each template has a pandas array of samples
        else: 
            stats = evaluate(y_pred, y_true, eval_labels)
            results.append(stats)

    return results, y_pred

def make_true_labels(
    cxr_true_labels_path: str, 
    cxr_labels: List[str],
    cutlabels: bool = True
): 
    """
    Loads in data containing the true binary labels
    for each pathology in `cxr_labels` for all samples. This
    is used for evaluation of model performance.

    args: 
        * cxr_true_labels_path - str, path to csv containing ground truth labels
        * cxr_labels - List[str], subset of label columns to select from ground truth df 
        * cutlabels - bool, if True, will keep columns of ground truth labels that correspond
            with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.

    Returns a numpy array of shape (# samples, # labels/pathologies)
        representing the binary ground truth labels for each pathology on each sample.
    """
    # create ground truth labels
    full_labels = pd.read_csv(cxr_true_labels_path)
    if cutlabels: 
        full_labels = full_labels.loc[:, cxr_labels]
    else: 
        full_labels.drop(full_labels.columns[0], axis=1, inplace=True)

    y_true = full_labels.to_numpy()
    return y_true

def make(
    model_path: str, 
    cxr_filepath: str, 
    pretrained: bool = True, 
    context_length: bool = 77, 
    change_text_encoder: bool = False,
):
    """
    FUNCTION: make
    -------------------------------------------
    This function makes the model, the data loader, and the ground truth labels. 
    
    args: 
        * model_path - String for directory to the weights of the trained clip model. 
        * context_length - int, max number of tokens of text inputted into the model. 
        * cxr_filepath - String for path to the chest x-ray images. 
        * cxr_labels - Python list of labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * pretrained - bool, whether or not model uses pretrained clip weights
        * cutlabels - bool, if True, will keep columns of ground truth labels that correspond
        with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.
    
    Returns model, data loader. 
    """
    # load model
    model = load_clip(
        model_path=model_path, 
        pretrained=pretrained, 
        context_length=context_length,
        change_text_encoder=change_text_encoder,
    )

    # load data
    transformations = [
        # means computed from sample in `cxr_stats` notebook
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    ]
    # if using CLIP pretrained model
    if pretrained: 
        # resize to input resolution of pretrained clip model
        input_resolution = 224
        transformations.append(Resize(input_resolution, interpolation=InterpolationMode.BICUBIC))
    transform = Compose(transformations)
    
    # create dataset
    torch_dset = CXRTestDataset(
        img_path=cxr_filepath,
        transform=transform, 
    )
    loader = torch.utils.data.DataLoader(torch_dset, shuffle=False)
    
    return model, loader

## Run the model on the data set using ensembled models
def ensemble_models(
    model_paths: List[str], 
    cxr_filepath: str, 
    cxr_labels: List[str], 
    cxr_pair_template: Tuple[str], 
    cache_dir: str = None, 
    save_name: str = None,
) -> Tuple[List[np.ndarray], np.ndarray]: 
    """
    Given a list of `model_paths`, ensemble model and return
    predictions. Caches predictions at `cache_dir` if location provided.

    Returns a list of each model's predictions and the averaged
    set of predictions.
    """

    predictions = []
    model_paths = sorted(model_paths) # ensure consistency of 
    for path in model_paths: # for each model
        model_name = Path(path).stem

        # load in model and `torch.DataLoader`
        model, loader = make(
            model_path=path, 
            cxr_filepath=cxr_filepath, 
        ) 
        
        # path to the cached prediction
        if cache_dir is not None:
            if save_name is not None: 
                cache_path = Path(cache_dir) / f"{save_name}_{model_name}.npy"
            else: 
                cache_path = Path(cache_dir) / f"{model_name}.npy"

        # if prediction already cached, don't recompute prediction
        if cache_dir is not None and os.path.exists(cache_path): 
            print("Loading cached prediction for {}".format(model_name))
            y_pred = np.load(cache_path)
        else: # cached prediction not found, compute preds
            print("Inferring model {}".format(path))
            y_pred = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template)
            if cache_dir is not None: 
                Path(cache_dir).mkdir(exist_ok=True, parents=True)
                np.save(file=cache_path, arr=y_pred)
        predictions.append(y_pred)
    
    # compute average predictions
    y_pred_avg = np.mean(predictions, axis=0)
    
    return predictions, y_pred_avg

def run_zero_shot(cxr_labels, cxr_templates, model_path, cxr_filepath, final_label_path, alt_labels_dict: dict = None, softmax_eval = True, context_length=77, pretrained: bool = False, use_bootstrap=True, cutlabels=True): 
    """
    FUNCTION: run_zero_shot
    --------------------------------------
    This function is the main function to run the zero-shot pipeline given a dataset, 
    labels, templates for those labels, ground truth labels, and config parameters.
    
    args: 
        * cxr_labels - list
            labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
            task can either be a string or a tuple (name of alternative label, name of label in csv)
        * cxr_templates - list, phrases that will be indpendently tested as input to the clip model. If `softmax_eval` is True, this parameter should be a 
        list of positive and negative template pairs stored as tuples. 
        * model_path - String for directory to the weights of the trained clip model. 
        * cxr_filepath - String for path to the chest x-ray images. 
        * final_label_path - String for path to ground truth labels.

        * alt_labels_dict (optional) - dict, map cxr_labels to list of alternative labels (i.e. 'Atelectasis': ['lung collapse', 'atelectatic lung', ...]) 
        * softmax_eval (optional) - bool, if True, will evaluate results through softmax of pos vs. neg samples. 
        * context_length (optional) - int, max number of tokens of text inputted into the model. 
        * pretrained (optional) - bool, whether or not model uses pretrained clip weights
        * use_bootstrap (optional) - bool, whether or not to use bootstrap sampling
        * cutlabels (optional) - bool, if True, will keep columns of ground truth labels that correspond
        with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining. 
    
    Returns an array of results per template, each consists of a tuple containing a pandas dataframes 
    for n bootstrap samples, and another pandas dataframe with the confidence intervals for each class.
    """
    
    np.random.seed(97)
    # make the model, data loader, and ground truth labels
    model, loader = make(
        model_path=model_path, 
        cxr_filepath=cxr_filepath, 
        pretrained=pretrained,
        context_length=context_length
    )

    y_true = make_true_labels(
        cxr_true_labels_path=final_label_path, 
        cxr_labels=cxr_labels, 
        cutlabels=cutlabels,
    )

    # run multiphrase experiment
    results, y_pred = run_experiment(model, cxr_labels, cxr_templates, loader, y_true, 
                             alt_labels_dict=alt_labels_dict, softmax_eval=softmax_eval, context_length=context_length, use_bootstrap=use_bootstrap)
    return results, y_pred

def run_cxr_zero_shot(model_path, context_length=77, pretrained=False): 
    """
    FUNCTION: run_cxr_zero_shot
    --------------------------------------
    This function runs zero-shot specifically for the cxr dataset. 
    The only difference between this function and `run_zero_shot` is that
    this function is already pre-parameterized for the 14 cxr labels evaluated
    using softmax method of positive and negative templates. 
    
    args: 
        * model_path - string, filepath of model being evaluated
        * context_length (optional) - int, max number of tokens of text inputted into the model. 
        * pretrained (optional) - bool, whether or not model uses pretrained clip weights
        * use_bootstrap (optional) - bool, whether or not to use bootstrap sampling
    
    Returns an array of labels, and an array of results per template, 
    each consists of a tuple containing a pandas dataframes 
    for n bootstrap samples, and another pandas dataframe with the confidence intervals for each class.
    """
    cxr_filepath = '/deep/group/data/med-data/test_cxr.h5'
    final_label_path = '/deep/group/data/med-data/final_paths.csv'
    
    cxr_labels = ['Atelectasis','Cardiomegaly', 
                                      'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                      'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                      'Pneumothorax', 'Support Devices']

    # templates list of positive and negative template pairs
    cxr_templates = [("{}", "no {}")]
    
    cxr_results = run_zero_shot(cxr_labels, cxr_templates, model_path, cxr_filepath=cxr_filepath, final_label_path=final_label_path, softmax_eval=True, context_length=context_length, pretrained=pretrained, use_bootstrap=False, cutlabels=True)
    
    return cxr_labels, cxr_results[0]

def validation_zero_shot(model_path, context_length=77, pretrained=False): 
    """
    FUNCTION: validation_zero_shot
    --------------------------------------
    This function uses the CheXpert validation dataset to make predictions
    on an alternative task (ap/pa, sex) in order to tune hyperparameters.
    
    args: 
        * model_path - string, filepath of model being evaluated
        * context_length (optional) - int, max number of tokens of text inputted into the model. 
        * pretrained (optional) - bool, whether or not model uses pretrained clip weights
        * use_bootstrap (optional) - bool, whether or not to use bootstrap sampling
    
    Returns an array of labels, and an array of results per template, 
    each consists of a tuple containing a pandas dataframes 
    for n bootstrap samples, and another pandas dataframe with the confidence intervals for each class.
    """
    cxr_sex_labels = ['Female', 'Male']

    cxr_sex_templates = [
                        #'{}', 
    #                      'the patient is a {}',
                         "the patient's sex is {}",
                        ]
    
    # run zero shot experiment
    sex_labels_path = '../../data/val_sex_labels.csv'
    results = run_zero_shot(cxr_sex_labels, cxr_sex_templates, model_path, cxr_filepath=cxr_filepath, final_label_path=sex_labels_path, softmax_eval=False, context_length=context_length, pretrained=True, use_bootstrap=True, cutlabels=False)
   
    results = run_zero_shot(cxr_sex_labels, cxr_sex_templates, model_path, cxr_filepath=cxr_filepath, final_label_path=sex_labels_path, softmax_eval=False, context_length=context_length, pretrained=True, use_bootstrap=True, cutlabels=False)
    pass


    
    
    
    
    
 
