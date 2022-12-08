import argparse
import cv2
import numpy as np
import torch
import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer

from train import train_main, load_data, load_clip, preprocess_text, preprocess_text_bert
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


def reshape_transform(tensor, height=7, width=7):
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

    model_clip = load_clip(model_path=None, pretrained=True, context_length=77).float().cpu()
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

            p = '/home/ec2-user/CHEXLOCALIZE/CheXpert/out_val/'+ os.path.join(*image_path.split('/')[-3:]).replace('/','_').replace('.jpg','_'+str(image_caption)+'_map.pkl')
            with open(p, 'wb') as handle:
                pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
         #   print(p)