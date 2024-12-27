from importlib_metadata import re
import torch
import numpy as np
import transformers
import torchvision
from torchvision import models, transforms
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

device = torch.device("cuda")

class TextEncoder(nn.Module):

    def __init__(self, text_fc2_out=32, text_fc1_out=2742, dropout_p=0.4, fine_tune_module=False):

        super(TextEncoder, self).__init__()

        self.fine_tune_module = fine_tune_module

        self.bert = BertModel.from_pretrained(
                    'bert-base-uncased',
#                     output_attentions = True,
                    return_dict=True)

        self.text_enc_fc1 = torch.nn.Linear(768, text_fc1_out)

        self.text_enc_fc2 = torch.nn.Linear(text_fc1_out, text_fc2_out)

        self.dropout = nn.Dropout(dropout_p)

        self.fine_tune()

    def forward(self, input_ids, attention_mask):
        """

        input_ids (torch.Tensor): 输入 (batch_size,max_length)

        attention_mask (torch.Tensor): attention mask information (batch_size, max_length)

        logits (torch.Tensor): 输出 (batch_size, num_labels)
        """

        # BERT
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # print(out['pooler_output'].shape)
        x = self.dropout(
            torch.nn.functional.relu(
                self.text_enc_fc1(out['pooler_output']))
        )

        x = self.dropout(
            torch.nn.functional.relu(
                self.text_enc_fc2(x))
        )

        return x

    def fine_tune(self):
        """
        固定参数
        """
        for p in self.bert.parameters():
            p.requires_grad = self.fine_tune_module


# vgg19
class VisionEncoder(nn.Module):

    def __init__(self, img_fc1_out=2742, img_fc2_out=32, dropout_p=0.4, fine_tune_module=False):
        super(VisionEncoder, self).__init__()

        self.fine_tune_module = fine_tune_module

        vgg = models.vgg19(pretrained=True)
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:1])

        self.vis_encoder = vgg

        self.vis_enc_fc1 = torch.nn.Linear(4096, img_fc1_out)

        self.vis_enc_fc2 = torch.nn.Linear(img_fc1_out, img_fc2_out)

        self.dropout = nn.Dropout(dropout_p)

        self.fine_tune()

    def forward(self, images):
        """
        : images, tensor (batch_size, 3, image_size, image_size)
        : encoded images
        """

        x = self.vis_encoder(images)

        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc1(x))
        )

        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc2(x))
        )

        return x

    def fine_tune(self):
    
        for p in self.vis_encoder.parameters():
            p.requires_grad = False

        for c in list(self.vis_encoder.children())[5:]:
            for p in c.parameters():
                p.requires_grad = self.fine_tune_module

#LanguageAndVisionConcat
class Text_Concat_Vision(torch.nn.Module):

    def __init__(self,
        model_params
    ):
        super(Text_Concat_Vision, self).__init__()

        self.text_encoder = TextEncoder(model_params['text_fc2_out'], model_params['text_fc1_out'], model_params['dropout_p'], model_params['fine_tune_text_module'])
        self.vision_encode = VisionEncoder(model_params['img_fc1_out'], model_params['img_fc2_out'], model_params['dropout_p'], model_params['fine_tune_vis_module'])

        self.fusion = torch.nn.Linear(
            in_features=(model_params['text_fc2_out'] + model_params['img_fc2_out']),
            out_features=model_params['fusion_output_size']
        )
        self.fc = torch.nn.Linear(
            in_features=model_params['fusion_output_size'],
            out_features=1
        )
        self.dropout = torch.nn.Dropout(model_params['dropout_p'])

        self.alpha = torch.nn.Linear(
            in_features=model_params['fusion_output_size'],
            out_features=model_params['fusion_output_size']
        )

        self.mu = torch.nn.Linear(
            in_features=model_params['fusion_output_size'],
            out_features=model_params['fusion_output_size']
        )

        self.logvar = torch.nn.Linear(
            in_features=model_params['fusion_output_size'],
            out_features=model_params['fusion_output_size']
        )



    def reparameterize(self, alpha, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        alpha_smoothed = eps * std + mu + alpha
        #z =  F.softmax((rho + alpha), dim=1)
        #rho =  F.softplus(eps * std + mu)
        rho =  F.softmax(alpha_smoothed, dim=1)
        #rho = eps * std + mu
        # Sample from standard Gaussian
        #rho = torch.rand_like(alpha)
        #z = torch.exp(torch.log(alpha) + rho)
        #z /= z.sum(-1, keepdim=True)
        #print(f"The value of Z is : {z}")
        #alpha_smoothed = rho + alpha
        #z = Dirichlet(alpha_smoothed.exp().cpu())

        return rho, alpha_smoothed


    #def forward(self, text, image, label=None):
    def forward(self, text, image, label=None):

        ## text to Bert
        text_features = self.text_encoder(text[0], text[1])
        ## image to vgg
        image_features = self.vision_encode(image)

        combined_features = torch.cat(
            [text_features, image_features], dim = 1
        )

        #print(f"combined_features: {combined_features.shape}")

        combined_features = self.dropout(combined_features)

        fused_ = self.dropout(
            torch.relu(
            self.fusion(combined_features)
            )
        )

        alpha = self.alpha(fused_)
        mu = self.mu(fused_)
        logvar = self.logvar(fused_)

        fused, alpha_smoothed = self.reparameterize(alpha, mu, logvar)
        #print(f"fused: {fused.shape}")
        #fused = fused + fused_
        fused = 0.4 * fused_ + 0.6 * fused

        # prediction = torch.nn.functional.sigmoid(self.fc(fused))
        prediction = torch.sigmoid(self.fc(fused))

        prediction = prediction.squeeze(-1)

        # prediction = prediction.cpu().detach().numpy()

        # for i in range(len(prediction)):
        #     if prediction[i] > 0.5:
        #         prediction[i] = 1.0
        #     else:
        #         prediction[i] = 0.0

        # prediction = torch.tensor(prediction, dtype=torch.float32, requires_grad = False).to(device)
        prediction = prediction.float()

        return prediction, alpha_smoothed