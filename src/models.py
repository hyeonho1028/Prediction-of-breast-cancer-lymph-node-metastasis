import timm

import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.config = args
        chans = 3
        num_classes = self.config.data.num_classes

        self.model = timm.create_model(model_name='tf_efficientnet_b0_ns', pretrained=True, in_chans=chans)
        # tf_efficientnet_b0_ns, efficientnet_b1_pruned, vit_base_patch16_224, tf_efficientnetv2_s_in21k

        if hasattr(self.model, "fc"):
            nb_ft = self.model.fc.in_features
            # self.model.fc = nn.Linear(nb_ft, num_classes)
            self.model.fc = nn.Identity()
        elif hasattr(self.model, "_fc"):
            nb_ft = self.model._fc.in_features
            # self.model._fc = nn.Linear(nb_ft, num_classes)
            self.model._fc = nn.Identity()
        elif hasattr(self.model, "classifier"):
            nb_ft = self.model.classifier.in_features
            # self.model.classifier = nn.Linear(nb_ft, num_classes)
            self.model.classifier = nn.Identity()
        elif hasattr(self.model, "last_linear"):
            nb_ft = self.model.last_linear.in_features
            # self.model.last_linear = nn.Linear(nb_ft, num_classes)
            self.model.last_linear = nn.Identity()
        elif hasattr(self.model, "head"):
            nb_ft = self.model.head.in_features
            # self.model.last_linear = nn.Linear(nb_ft, num_classes)
            self.model.head = nn.Identity()    
        
        self.fc = nn.Linear(nb_ft, num_classes)
        
        # self.fc = nn.Sequential(
        #                             nn.Linear(512, num_classes),
        #                             # nn.ReLU(num_classes),
        #                         )

    def forward(self, x):
        x = self.model(x)
        output = self.fc(x)
        return output