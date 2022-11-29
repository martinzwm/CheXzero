import os
import sys
sys.path.append('/opt/conda/envs/biovil/lib/python3.9/site-packages') # add path to biovil text encoder

import pprint
import argparse
from tqdm import tqdm

import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, Resize

import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer

from train import train_main, load_data, load_clip, preprocess_text
from zero_shot import run_cxr_zero_shot, run_zero_shot

from accelerate import Accelerator
from visualize import *

# cxr bert encoder
from health_multimodal.text.utils import get_cxr_bert, get_cxr_bert_inference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='data/backup/cxr.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--txt_filepath', type=str, default='data/mimic_impressions.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default="checkpoints/", help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--model_name', type=str, default="pt-imp")
    args = parser.parse_args()
    return args

def model_pipeline(config, verbose=0): 
    # make the model, data, and optimization problem
    model, data_loader, device, criterion, optimizer = make(config, cxr_bert=False)

    # and use them to train the model
    train(model, data_loader, device, criterion, optimizer, config, cxr_bert=False)

    # save model
    model_path = os.path.join(config.save_dir, str(config.model_name), 'checkpoint.pt')
    save(model, model_path)

    if verbose: 
        print(model)
    return model

@make_wandb
def make(config, cxr_bert=False): 
    pretrained = not config.random_init
    data_loader, device = load_data(config.cxr_filepath, config.txt_filepath, batch_size=config.batch_size, pretrained=pretrained, column="impression")
    model = load_clip(model_path=None, pretrained=pretrained, context_length=config.context_length)
    
    # modify model internal to use bert text encoder
    if cxr_bert:
        tokenizer, text_model = get_cxr_bert()
        model.text_model = text_model
        model.text_model_l1 = nn.Linear(128, 512)
        model.text_model_l2 = nn.Linear(512, 512)
    
    model.to(device)
    print('Model on Device.')

    # make the optimizer 
    criterion = nn.CrossEntropyLoss().cuda()
    if config.optimizer == "adam": 
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer == "sgd": 
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    return model, data_loader, device, criterion, optimizer

def train(model, loader, device, criterion, optimizer, config, cxr_bert=False):
    # Multi-GPU
    accelerator = Accelerator()
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    model_save_dir = os.path.join(config.save_dir, config.model_name)
    if not os.path.exists(model_save_dir): 
        # Create a new folder if not exists
        os.makedirs(model_save_dir)
    
    # For cxr bert text encoder
    text_inference = get_cxr_bert_inference()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run training
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    report_freq = config.log_interval
    highest_val_auc = 0 # save highest mean auc
    
    for epoch in range(config.epochs):
        running_loss = 0.0 # running loss over batch
        for data in tqdm(loader):
            # get the images
            images = data['img']

            texts = data['txt']

            # preprocess text
            if not cxr_bert:
                texts = preprocess_text(texts, model)
                attention_mask = None
            else:
                tokenizer_output = text_inference.tokenize_input_prompts(texts)
                texts = tokenizer_output['input_ids'].to(accelerator.device)
                attention_mask = tokenizer_output['attention_mask'].to(accelerator.device)

            texts = texts.to(accelerator.device)
            
            # perform step for a single batch
            loss = train_batch(images, texts, model, device, criterion, optimizer, accelerator, attention_mask=attention_mask)
            example_ct +=  len(images)
            batch_ct += 1
            running_loss += loss.item()

            # Report metrics every `report_freq` batch
            if (batch_ct % report_freq) == 0:
                train_log(running_loss / report_freq, example_ct, epoch)
                running_loss = 0.0
            
            if (batch_ct % config.save_interval) == 0: 
                model_path = os.path.join(model_save_dir, "checkpoint_{batch_ct}.pt".format(
                    batch_ct=str(batch_ct), 
                ))
                print("Saved checkpoint to: ", model_path)
                save(model, model_path)
                
def train_batch(images, texts, model, device, criterion, optimizer, accelerator, attention_mask=None):
    # images, texts = images.to(device), texts.to(device)

    # Forward pass
    logits_per_image, logits_per_text = model(images, texts, attention_mask=attention_mask)
    
    # Create labels
    batch_size = images.shape[0]
    labels = torch.arange(batch_size).to(device)
    
    # Compute loss
    loss_img = criterion(logits_per_image, labels)
    loss_txt = criterion(logits_per_text, labels)
    loss = (loss_img + loss_txt)/2 # avg. img and txt loss

    # Backward pass ⬅
    optimizer.zero_grad()
    # loss.backward()
    accelerator.backward(loss)
    
    # Step with optimizer
    optimizer.step()
        
    return loss

@train_log_wandb
def train_log(loss, example_ct, epoch):
    loss = float(loss)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

@save_wandb
def save(model, path): 
    torch.save(model.state_dict(), path)
    
if __name__ == "__main__":
    torch.manual_seed(123)
    args = parse_args()
    model = model_pipeline(args)
    

