import argparse
import cv2
import numpy as np
import torch
import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer

from train import train_main, load_data, preprocess_text, preprocess_text_bert, load_clip
from zero_shot import run_cxr_zero_shot, run_zero_shot
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import clip
from PIL import Image
from scipy.ndimage import filters
from torch import nn
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map

def viz_attn(img, attn_map, blur=True):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[1].imshow(getAttMap(img, attn_map, blur))
    for ax in axes:
        ax.axis("off")
    plt.show()
    
def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.
class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='/home/ec2-user/CheXzero-chenwei/data/MIMIC-CXR-JPG/raw/files/p12/p12533192/s50682827/eedf55e3-582abefc-1da7ceae-51c7da73-e3826cc5.jpg',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

def load_clip_model(model_path, pretrained=False, context_length=77, change_text_encoder=False): 
    """
    FUNCTION: load_clip
    ---------------------------------
    """
    device = torch.device("cuda:0")
    if pretrained is False: 
        # use new model params
        params = {
            'embed_dim':768,
            'image_resolution': 224,
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
        # if change_text_encoder:
        tokenizer, text_model = get_cxr_bert()
        model.text_model = text_model
        model.text_model_l1 = nn.Linear(128, 512)
        model.text_model_l2 = nn.Linear(512, 512)
        try: 
            print(model_path,"using an online model  ")
            # model=torch.load(model_path).to(device)
            model.load_state_dict(torch.load(model_path))
            model = model.to(device)
        except: 
            print("Argument error. Set pretrained = True.", sys.exc_info()[0])
            raise
    return model
def reshape_transform(tensor, height=14, width=14):
  #  print(tensor.shape)
    tensor = tensor.permute(1,0,2)
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
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

 

def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:        
        # Do a forward and backward pass.
      #  input= reshape_transform(input)
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()
    
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam
if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    # model = torch.hub.load('facebookresearch/deit:main',
    #                        'deit_tiny_patch16_224', pretrained=True)
    # model.eval()
    cxr_filepath: str = '/home/ec2-user/CHEXLOCALIZE/CheXpert/test.h5'
    model_clip = load_clip_model(model_path='/home/ec2-user/CheXzero/checkpoints/cxr-bert/checkpoint_45000.pt', pretrained=False, context_length=77).float().cpu()
    model = model_clip
    model.eval()

    print('Model on Device.')
    #print(model)
    target_layers = [model.visual.transformer.resblocks[10].ln_2]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform)



    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.





    #-------------------
     #@param {type:"string"}
    # #@markdown ---
    # blur = True #@param {type:"boolean"}

    # device = "cuda" if torch.cuda.is_available() else "cpu"
  #  model, preprocess = clip.load(clip_model, device=device, jit=False)
 #   print(text.shape)
    # Download the image from the web.
   # text_input = model_clip.tokenize([image_caption]).to(device)
    # image_caption = preprocess_text(image_caption, model_clip)
    # attn_map = gradCAM(
    #     model,
    #     input_tensor,
    #     model_clip.encode_text(image_caption.cuda()).float(),
    #     layer=model.transformer.resblocks[10].ln_2,
    # )
    # attn_map = attn_map.squeeze().detach().cpu().numpy()
    #-------------------preprocess text
    # print(targets.shape)
    # _tokenizer = SimpleTokenizer()
    # sot_token = _tokenizer.encoder["<|startoftext|>"]
    # eot_token = _tokenizer.encoder["<|endoftext|>"]
    # all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    # result = torch.zeros(len(all_tokens), model.context_length, dtype=torch.long)
    
    # for i, tokens in enumerate(all_tokens):
    #     if len(tokens) > model.context_length:
    #         tokens = tokens[:model.context_length]
    #         tokens[model.context_length - 1] = eot_token
    #     result[i, :len(tokens)] = torch.tensor(tokens)
    

    classnames = [ 'Lung Opacity',
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Enlarged Cardiomediastinum',
    'Lung Lesion',
    'Pleural Effusion',
    'Pneumothorax',
    'Support Devices',
    ]

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32
    import glob
    import pickle
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
    all_paths = glob.glob('/home/ec2-user/CHEXLOCALIZE/CheXpert/valid/*/*/*.jpg')
    metadata = pd.read_csv('/home/ec2-user/CHEXLOCALIZE/CheXpert/val_labels.csv')
 #   print(all_paths)
    for image_path in tqdm(all_paths,total=len(all_paths)):
        print(image_path)
        df = metadata.loc[metadata['cur_path'] == image_path]
        for image_caption in classnames: 
            
            rgb_img = cv2.imread(image_path)[:, :, ::-1]
            w,h,c = rgb_img.shape
            rgb_img = cv2.resize(rgb_img, (224, 224))
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
            input_tensor = input_tensor.float()
        
            targets = classnames.index(image_caption)
            
            text = preprocess_text(image_caption, model_clip)                    
            templates: Tuple[str] = ("{}", "no {}")

            zeroshot_weights = []
            # compute embedding through model for each class\
            with torch.no_grad():
                for classname in classnames:
                    texts = [template.format(classname) for template in templates] # format with class
                    texts = clip.tokenize(texts, context_length=77) # tokenize
                    texts = texts
                    
                    class_embeddings = model.encode_text(texts) # embed with text encoder
                    
                    # normalize class_embeddings
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    # average over templates 
                    class_embedding = class_embeddings.mean(dim=0) 
                    # norm over new averaged templates
                    class_embedding /= class_embedding.norm() 
                    zeroshot_weights.append(class_embedding)
                zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
            with torch.autograd.set_detect_anomaly(True):
                grayscale_cam = cam(input_tensor=(input_tensor,text,zeroshot_weights),
                                targets=ClassifierOutputSoftmaxTarget(targets),
                                eigen_smooth=args.eigen_smooth,
                                aug_smooth=args.aug_smooth)
            output={}
            probs = model((input_tensor,text,zeroshot_weights))
            probability = nn.functional.softmax(probs, dim=0)[targets].item()
            grayscale_cam = grayscale_cam[0, :]
            cxr_img = input_tensor.squeeze(0)
            map = torch.from_numpy(grayscale_cam).unsqueeze(0).unsqueeze(0)
            gt = df[image_caption].values[0]
            task = image_caption
            cxr_dims = (w,h)
            output['cxr_img'] = cxr_img
            output['map'] = map
            output['prob'] = probability
            output['gt'] = gt
            output['task'] = task
            output['cxr_dims'] = cxr_dims
            print(probability,cxr_dims,task,gt,cxr_img.shape,map.shape)

            p = '/home/ec2-user/CHEXLOCALIZE/CheXpert/fuse_out_val/'+ os.path.join(*image_path.split('/')[-3:]).replace('/','_').replace('.jpg','_'+str(image_caption)+'_map.pkl')
            with open(p, 'wb') as handle:
                pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
         #   print(p)