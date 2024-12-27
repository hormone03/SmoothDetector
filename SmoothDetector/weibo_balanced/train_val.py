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
from sklearn.metrics import classification_report


def train(model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader=None, epochs=4, evaluation=False, device='cpu',
            param_dict_model=None, param_dict_opt=None, save_best=False, file_path='saved_models/best_model.pt',
            writer=None
            ):

    # training_loop
    best_acc_val = 0
    #z_latent = []
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================


        print(f"{'Epoch':^3} {'Batch':^3}| {'cls_loss':^12} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*80)

        t0_epoch, t0_batch = time.time(), time.time()

        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            batch_counts +=1

            img_ip , text_ip, label = batch["image"], batch["BERT_ids_mask"], batch['label']

            b_input_ids, b_attn_mask = tuple(t.to(device) for t in text_ip)

            imgs_ip = img_ip.to(device)


            b_labels = label.to(device)


            model.zero_grad()


            # logits, att_mask_img = model(text=[b_input_ids, b_attn_mask], image=imgs_ip, label=b_labels)
            logits, z = model(text=[b_input_ids, b_attn_mask], image=imgs_ip)
            
            #z_latent.append(z.mean(dim=0).cpu().detach().numpy())
            #if step % 5 ==0:
                #z_latent_mean = np.array(z_latent).mean(axis=0)
                #print(z_latent_mean)
            b_labels=b_labels.to(torch.float32)
            loss, cls_loss = loss_fn(logits, b_labels, z)
            batch_loss += loss.item()
            total_loss += loss.item()
#####################################################################
            #print("_____" * 5)
            #label_ = b_labels.cpu().detach().numpy()
            #logits_copy = logits.cpu().detach().numpy()
            #logits_copy[logits_copy<0.5] = 0
            #logits_copy[logits_copy>=0.5] = 1



           # report = classification_report(logits_copy, label_, output_dict=True)

            #print(f"loss: {loss}")
           # print(f"b_labels: {b_labels}")
            #print(f"logits_copy: {logits_copy}")
            #print(report)
            #print("_____" * 5)
#####################################################################
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if (step % 100 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch

                print(f"{epoch_i + 1:^3} : {step:^3}  | {cls_loss/batch_counts:^12.6f} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                if writer != None:
                    writer.add_scalar('Training Loss', (batch_loss / batch_counts), epoch_i*len(train_dataloader)+step)

                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*80)

        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            test_t = time.time()
            val_loss, val_accuracy, report = evaluate(model, loss_fn, val_dataloader, device)

            time_elapsed = test_t - time.time()
            print(f" {epoch_i + 1:^6} | {'-':^12} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*80)
            print("tsting time: " + str(time_elapsed))
            print("-"*80)
            print(report)
            print("-"*80)

            if writer != None:
                writer.add_scalar('Validation Loss', val_loss, epoch_i+1)
                writer.add_scalar('Validation Accuracy', val_accuracy, epoch_i+1)

            # best_model
            if save_best:
                if val_accuracy > best_acc_val:
                    best_acc_val = val_accuracy
                    torch.save({
                                'epoch': epoch_i+1,
                                'model_params': param_dict_model,
                                'opt_params': param_dict_opt,
                                'model_state_dict': model.state_dict(),
                                'opt_state_dict': optimizer.state_dict(),
                                'sch_state_dict': scheduler.state_dict()
                               }, file_path)

        print("\n")

    print("Training complete!")



def evaluate(model, loss_fn, val_dataloader, device):
    """
        在每个epoch训练完成后，测试模型的性能
    """

    model.eval()

    val_accuracy = []
    val_loss = []
    final_out = []
    final_lab = []
    z_latent = []

    for batch in val_dataloader:
        img_ip , text_ip, label = batch["image"], batch["BERT_ids_mask"], batch['label']

        b_input_ids, b_attn_mask = tuple(t.to(device) for t in text_ip)

        imgs_ip = img_ip.to(device)

        b_labels = label.to(device)


        with torch.no_grad():
            # logits, att_mask_img = model(text=[b_input_ids, b_attn_mask], image=imgs_ip, label=b_labels)
            logits, z = model(text=[b_input_ids, b_attn_mask], image=imgs_ip)
            b_labels=b_labels.to(torch.float32)

        z_latent.append(z.mean(dim=0).cpu().detach().numpy())

        loss, cls_loss = loss_fn(logits, b_labels, z)
        val_loss.append(loss.item())

        logits[logits<0.5] = 0
        logits[logits>=0.5] = 1

        label_ = b_labels.cpu().detach().numpy()
        logits_copy = logits.cpu().detach().numpy()
        final_out.extend(list(logits_copy))
        final_lab.extend(list(label_))
        # print(logits)

        # preds = torch.argmax(logits, dim=1).flatten()
        #print(preds)
        accuracy = (logits == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    report = classification_report(final_lab, final_out, output_dict=True)
    print('*****' * 4)
    z_latent_mean = np.array(z_latent).mean(axis=0)
    print(z_latent_mean)
    print('*****' * 4)
    return val_loss, val_accuracy, report