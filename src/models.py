import timm

import torch
import torch.nn as nn


class Str_Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.config = args
        self.categorical_embeddings = nn.Embedding(
            sum(args.train_params.cat_features_ls), args.embedding_size
            )
        self.numerical_direction = nn.Parameter(
            torch.rand(args.train_params.num_numeric_features, args.embedding_size)
        )
        self.numerical_anchor = nn.Parameter(
            torch.rand(args.train_params.num_numeric_features, args.embedding_size)
        )

        self.register_buffer(
            "categorical_embedding_offsets",
            torch.tensor([[0] + args.train_params.cat_features_ls[:-1]]).cumsum(1),
        )

    def forward(self, cat_features, num_features):
        cat_inputs = cat_features + self.categorical_embedding_offsets
        cat_embeddings = self.categorical_embeddings(cat_inputs)

        num_embeddings = num_features[:, :, None] * self.numerical_direction
        num_embeddings = num_embeddings + self.numerical_anchor
        
        return torch.cat(
            (cat_embeddings, num_embeddings), dim=1
        )


class ImageVectorizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.config = args
        chans = 3
        self.model = timm.create_model(model_name='tf_efficientnetv2_s_in21k', pretrained=True, in_chans=chans)
        # tf_efficientnet_b0_ns, efficientnet_b1_pruned, vit_base_patch16_224, tf_efficientnetv2_s_in21k

        if hasattr(self.model, "fc"):
            nb_ft = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif hasattr(self.model, "_fc"):
            nb_ft = self.model._fc.in_features
            self.model._fc = nn.Identity()
        elif hasattr(self.model, "classifier"):
            nb_ft = self.model.classifier.in_features
            self.model.global_pool = nn.Identity()
            self.model.classifier = nn.Identity()
        elif hasattr(self.model, "last_linear"):
            nb_ft = self.model.last_linear.in_features
            self.model.last_linear = nn.Identity()
        elif hasattr(self.model, "head"):
            nb_ft = self.model.head.in_features
            self.model.head = nn.Identity()

        self.linaer = nn.Linear(1280, args.embedding_size)
        
    def forward(self, x):
        bs = x.size(0)
        img_vector = self.model(x)
        out = self.linaer(img_vector.permute(0,2,3,1)).view(bs, -1, self.config.embedding_size)
        return out

class BreastCancerModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.img_vectorizer = ImageVectorizer(args)
        self.str_embedding = Str_Embedding(args)

        self.fc = nn.Linear(args.embedding_size*72, args.data.num_classes)
        
    def forward(self, img, cat_features, num_features):
        img_vector = self.img_vectorizer(img)
        str_vector = self.str_embedding(cat_features, num_features)

        concat = torch.cat([img_vector, str_vector], dim=1).flatten(1)
        output = self.fc(concat)
        return output






class CNNModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.config = args
        chans = 3
        num_classes = args.data.num_classes

        self.model = timm.create_model(model_name='tf_efficientnet_b4_ns', pretrained=True, in_chans=chans)
        # tf_efficientnet_b0_ns, efficientnet_b1_pruned, vit_base_patch16_224, tf_efficientnetv2_s_in21k

        if hasattr(self.model, "fc"):
            nb_ft = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif hasattr(self.model, "_fc"):
            nb_ft = self.model._fc.in_features
            self.model._fc = nn.Identity()
        elif hasattr(self.model, "classifier"):
            nb_ft = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif hasattr(self.model, "last_linear"):
            nb_ft = self.model.last_linear.in_features
            self.model.last_linear = nn.Identity()
        elif hasattr(self.model, "head"):
            nb_ft = self.model.head.in_features
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