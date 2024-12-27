import torch
import pandas as pd
import numpy as np
import transformers
import torchvision
from torchvision import models, transforms
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time
import os
import re
import math
from torch.distributions import Dirichlet, kl_divergence
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from models import *
from data_loader import *
from train_val import *


reshuffle_data = True
if reshuffle_data:
    df_train = pd.read_csv("twitter/train_posts_clean.csv")
    df_test = pd.read_csv("twitter/test_posts.csv")
    df = pd.concat([df_train, df_test], ignore_index=True, sort=False)
    oversample = RandomOverSampler(sampling_strategy='minority')
    y = df["label"]
    X = df.drop(["label"], axis=1)
    df_marged, y_over = oversample.fit_resample(X, y)
    df_marged['label'] = y_over
    df_minority = df_marged[df_marged['label']=='real']
    print(f"df_minority: {len(df_minority)}")
    df_majority = df_marged[df_marged['label']=='fake']
    print(f"df_majority: {len(df_majority)}")
    #df_majority = df_majority.sample(len(df_minority), random_state=0)
    #df_marged = pd.concat([df_majority, df_minority])
    #df_marged = df_marged.sample(frac=1, random_state=0)
    print(f"df_marged: {len(df_marged)}")
    df_train_s, df_test_s = train_test_split(df_marged, test_size=0.2, shuffle=True, random_state=0)
    df_train_s.to_csv('twitter/df_train2.csv', encoding='utf-8', index=False)
    df_test_s.to_csv('twitter/df_test2.csv', encoding='utf-8', index=False)
    df_train = pd.read_csv("twitter/df_train2.csv")
    df_test = pd.read_csv("twitter/df_test2.csv")

    print(f"length of training set: {len(df_train)}")
    print(f"length of test set: {len(df_test)}")

else:
    df_train = pd.read_csv("twitter/train_posts_clean.csv")
    df_test = pd.read_csv("twitter/test_posts.csv")
    df = pd.concat([df_train, df_test], ignore_index=True, sort=False)
    df_minority = df[df['label']=='real']
    print(f"df_minority: {len(df_minority)}")
    df_majority = df[df['label']=='fake']
    print(f"df_majority: {len(df_majority)}")

    df_train_s, df_test_s = train_test_split(df, test_size=0.2, shuffle=True, random_state=0)
    df_train_s.to_csv('twitter/df_train3.csv', encoding='utf-8', index=False)
    df_test_s.to_csv('twitter/df_test3.csv', encoding='utf-8', index=False)
    df_train = pd.read_csv("twitter/df_train3.csv")
    df_test = pd.read_csv("twitter/df_test3.csv")
    print(f"length of training set: {len(df_train)}")
    print(f"length of test set: {len(df_test)}")


#df_train = pd.read_csv("twitter/train_posts_clean.csv")
#print(f"length of training set: {len(df_train)}")
#df_minority = df_train[df_train['label']=='real']
#print(f"train_df_minority: {len(df_minority)}")
#df_majority = df_train[df_train['label']=='fake']
#print(f"train_df_majority: {len(df_majority)}")

#df_test = pd.read_csv("twitter/test_posts.csv")
#print(f"length of testing set: {len(df_test)}")
#df_minority = df_test[df_test['label']=='real']
#print(f"test_df_minority: {len(df_minority)}")
#df_majority = df_test[df_test['label']=='fake']
#print(f"test_df_majority: {len(df_majority)}")


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

MAX_LEN = 500
root_dir = "twitter/"

transformed_dataset_train = FakeNewsDataset(df_train, root_dir+"images_train/", image_transform, tokenizer, MAX_LEN)

transformed_dataset_val = FakeNewsDataset(df_test, root_dir+"images_test/", image_transform, tokenizer, MAX_LEN)

train_dataloader = DataLoader(transformed_dataset_train, batch_size=8,
                        shuffle=True, num_workers=0)

val_dataloader = DataLoader(transformed_dataset_val, batch_size=8,
                        shuffle=True, num_workers=0)


cls_BCE = nn.BCELoss()

def kld(model_alpha, prior_alpha, epsilon): #overturned

    model_alpha = torch.max(torch.tensor(0.0001), model_alpha).to(device)
    alpha = prior_alpha.expand_as(model_alpha)
    # model_alpha = F.softplus(model_alpha)  # To normalize between 0 and 1

    #log_ratio = torch.log(torch.digamma(model_alpha.prod()) / torch.digamma(alpha.prod()))
    #print(f"log_ratio: {log_ratio}")

    sum1 = torch.sum((model_alpha + epsilon - 1) * torch.digamma(model_alpha + epsilon), dim=1)
    #print(f"sum1: {sum1}")

    sum2 = torch.sum((alpha + epsilon - 1) * torch.digamma(alpha + epsilon), dim=1)
    #print(f"sum2: {sum2}")

    kl_loss = torch.mean(sum1 - sum2)

    return kl_loss #torch.tensor(kl_loss).to(device)

def kld__(model_alpha, prior_alpha, epsilon): #overturned

    model_alpha = torch.max(torch.tensor(0.0001), model_alpha).to(device)
    alpha = prior_alpha.expand_as(model_alpha)
    model_alpha = F.softplus(model_alpha)  # To normalize between 0 and 1

    #log_ratio = torch.log(torch.digamma(model_alpha.prod()) / torch.digamma(alpha.prod()))
    #print(f"log_ratio: {log_ratio}")

    sum1 = torch.sum((alpha + epsilon - 1) * torch.digamma(alpha + epsilon), dim=1)
    #print(f"sum1: {sum1}")

    sum2 = torch.sum((model_alpha + epsilon - 1) * torch.digamma(model_alpha + epsilon), dim=1)
    #print(f"sum2: {sum2}")

    kl_loss = torch.mean(sum1 - sum2)

    return kl_loss #torch.tensor(kl_loss).to(device)

def loss_fn(logits, b_labels, alpha_smoothed):

    cls_loss = cls_BCE(logits, b_labels)
    #print(cls_loss)

    #z = Dirichlet(alpha_smoothed)
    #prior_alpha = torch.ones_like(z.concentration) * 0.01
    #prior = Dirichlet(prior_alpha)
    #kld_loss = torch.sum(kl_divergence(z, prior).to(device))

    kld_loss = kld(alpha_smoothed, prior_alpha = torch.tensor(0.01), epsilon=torch.tensor(0.000000000001))
    #print(kld_loss)
    loss = cls_loss + kld_loss * 0.0001
    return loss, cls_loss

def set_seed(seed_value=42):
    """
        设置种子
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

parameter_dict_model={
    'text_fc2_out': 32,
    'text_fc1_out': 2742,
    'dropout_p': 0.4,
    'fine_tune_text_module': False,
    'img_fc1_out': 2742,
    'img_fc2_out': 32,
    'dropout_p': 0.4,
    'fine_tune_vis_module': False,
    'fusion_output_size': 35
}

parameter_dict_opt={'l_r': 3e-5,
                    'eps': 1e-8
                    }


EPOCHS = 20  # 10

set_seed(7)

final_model = Text_Concat_Vision(parameter_dict_model)

final_model = final_model.to(device)

optimizer = AdamW(final_model.parameters(),
                  lr=parameter_dict_opt['l_r'],
                  eps=parameter_dict_opt['eps'])

total_steps = len(train_dataloader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0, 
                                            num_training_steps=total_steps)
writer = SummaryWriter('multi_att_exp3')

train(model=final_model,
      loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
      train_dataloader=train_dataloader, val_dataloader=val_dataloader,
      epochs=1, evaluation=True, #epochs=150
      device=device,
      param_dict_model=parameter_dict_model, param_dict_opt=parameter_dict_opt,
      save_best=True,
      file_path='saved_models/best_model.pt'
      , writer=writer
      )