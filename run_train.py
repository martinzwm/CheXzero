import os
import sys
sys.path.append('/opt/conda/envs/biovil/lib/python3.9/site-packages') # add path to biovil text encoder

import pprint
import argparse
from tqdm import tqdm
import numpy as np
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

from visualize import *

from accelerate import Accelerator


from bidirectional_cross_attention import BidirectionalCrossAttention
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='data/backup/cxr.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--txt_filepath', type=str, default='data/mimic_impressions.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_interval', type=int, default=1000)
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
    model, data_loader, device, criterion, optimizer, joint_layer, linear= make(config)

    # and use them to train the model
    train(model, data_loader, device, criterion, optimizer, config, joint_layer = joint_layer, linear=linear)

    # save model
    model_path = os.path.join(config.save_dir, str(config.model_name), 'checkpoint.pt')
    save(model, model_path)

    if verbose: 
        print(model)
    return model

@make_wandb
def make(config): 
    pretrained = not config.random_init
    data_loader, device = load_data(config.cxr_filepath, config.txt_filepath, batch_size=config.batch_size, pretrained=pretrained, column="impression")
    model = load_clip(model_path=None, pretrained=pretrained, context_length=config.context_length)
    model.to(device)
    print('Model on Device.')

    joint_cross_attn = BidirectionalCrossAttention(
            dim = 512,
            heads = 8,
            dim_head = 64,
            context_dim = 512
        )
    linear = torch.nn.Sequential(
            nn.Linear(512,64),
            nn.Linear(64, 2))
    


    # make the optimizer 
    criterion = nn.CrossEntropyLoss().cuda()
    if config.optimizer == "adam": 
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer == "sgd": 
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    return model, data_loader, device, criterion, optimizer,joint_cross_attn, linear

def train(model, loader, device, criterion, optimizer, config, joint_layer, linear):
    # Multi-GPU
    accelerator = Accelerator()
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    model_save_dir = os.path.join(config.save_dir, config.model_name)
    if not os.path.exists(model_save_dir): 
        # Create a new folder if not exists
        os.makedirs(model_save_dir)
    
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
            texts = preprocess_text(texts, model)
            texts = texts.to(accelerator.device)
            
            # perform step for a single batch
            loss = train_batch(images, texts, model, device, criterion, optimizer, accelerator,joint_layer.to(accelerator.device), linear.to(accelerator.device))
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
                
def train_batch(images, texts, model, device, criterion, optimizer, accelerator, joint_layer, linear):
    # images, texts = images.to(device), texts.to(device)

    # Forward pass
    logits_per_image, logits_per_text, img_embed, txt_embed = model(images, texts)
    img_embed = img_embed.float().to(device)
    txt_embed = txt_embed.float().to(device)
    # Create labels
    batch_size = images.shape[0]
    labels = torch.arange(batch_size).to(device)
   # print(logits_per_image.shape, logits_per_text.shape)
    # Compute loss
    loss_img = criterion(logits_per_image, labels)
    loss_txt = criterion(logits_per_text, labels)
   # print(img_embed.shape, txt_embed.shape)
  #  print(img_embed.shape, txt_embed.shape)
    video_mask = torch.ones((1, img_embed.shape[0])).bool().cuda()
    text_mask = torch.ones((1, txt_embed.shape[0])).bool().cuda()
    img_out, txt_out = joint_layer(
        img_embed.unsqueeze(0),
        txt_embed.unsqueeze(0),
        mask = video_mask,
        context_mask = text_mask
    )
    cross_pos =  torch.concat(
        [img_out.squeeze(0),
        txt_out.squeeze(0) ], dim=0)
   # print("cross_pos",cross_pos.shape)
    joint_out = linear(cross_pos
    )
    with torch.no_grad():
        bs = images.size(0)          
        weights_i2t = torch.nan_to_num(torch.nn.functional.softmax(logits_per_image[:,:bs],dim=1),0)
        weights_t2i =torch.nan_to_num( torch.nn.functional.softmax(logits_per_text[:,:bs],dim=1),0)

        weights_i2t.fill_diagonal_(0)
        weights_t2i.fill_diagonal_(0)

    # select a negative image for each text
    image_embeds_neg = []    
    for b in range(bs):

        neg_idx = np.random.randint(0,bs)
        image_embeds_neg.append(img_embed[neg_idx])
    image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

    # select a negative text for each image
    text_embeds_neg = []
    for b in range(bs):
        neg_idx = np.random.randint(0,bs)
        text_embeds_neg.append(txt_embed[neg_idx])
    text_embeds_neg = torch.stack(text_embeds_neg,dim=0)     

    # text_embeds_all = torch.cat([txt_embed, text_embeds_neg],dim=0)         

    # image_embeds_all = torch.cat([image_embeds_neg,img_embed],dim=0)

    img_out_neg, txt_out_neg = joint_layer(
        image_embeds_neg.unsqueeze(0),
        text_embeds_neg.unsqueeze(0),
        mask = video_mask,
        context_mask = text_mask
    )                    

    cross_neg =  torch.concat(
        [img_out_neg.squeeze(0),
        txt_out_neg.squeeze(0) ], dim=0)
  #  print("cross_neg",cross_neg.shape)
    joint_out_neg = linear(cross_neg
    )         
    joint_out_all = torch.concat([joint_out,joint_out_neg],dim=0)
    itm_labels = torch.cat([torch.ones(2*bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                            dim=0).to(accelerator.device)
    criterion3 = torch.nn.CrossEntropyLoss()
    loss_itm = criterion3(joint_out_all, itm_labels)     
    
    
    loss = (loss_img + loss_txt + loss_itm)/3 # avg. img and txt loss


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
    

