import torch.nn as nn
from ltr import model_constructor

import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)

from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network
from .SEblock import SE_Block           # lpp
from .spectralSpatial import ChannelSpatialSELayer3D

import yaml
from yaml import CSafeLoader as Loader
import os

from ltr.models_.methods.SwinTrack.positional_encoding.builder import build_position_embedding
from ltr.models_.utils.drop_path import DropPathAllocator, DropPathScheduler
from ltr.models_.methods.SwinTrack.modules.encoder.builder import build_encoder
from ltr.models_.methods.SwinTrack.modules.decoder.builder import build_decoder
from ltr.models_.head.builder import build_head
from timm.models.layers import trunc_normal_


class TransT(nn.Module):
    """ This is the TransT module that performs single object tracking """
    
    def __init__(self, backbone, featurefusion_network, 
                 encoder_f_z, encoder_f_x, 
                 decoder_f_z,decoder_f_x,out_norm, out_norm_f_z,out_norm_f_x,
                 z_backbone_out_stage, x_backbone_out_stage,
                 z_input_projection, x_input_projection,
                 z_pos_enc, x_pos_enc, num_classes):

        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()

        self.num_channel = 1024
        
        self.featurefusion_network = featurefusion_network
        hidden_dim = featurefusion_network.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        self.backbone = backbone
        self.backbone = backbone

        self.encoder_f_z = encoder_f_z
        self.encoder_f_x = encoder_f_x

        self.decoder_f_z = decoder_f_z
        self.decoder_f_x = decoder_f_x
        # Relu lpp 2022/6/15
        self.relu = nn.ReLU(inplace=True)
        self.out_norm = out_norm
        self.out_norm_f_z = out_norm_f_z
        self.out_norm_f_x = out_norm_f_x
        
        self.z_backbone_out_stage = z_backbone_out_stage
        self.x_backbone_out_stage = x_backbone_out_stage
        self.z_input_projection = z_input_projection
        self.x_input_projection = x_input_projection

        self.z_pos_enc = z_pos_enc
        self.x_pos_enc = x_pos_enc
        self.reset_parameters()
        
        # # clk 22-12-19
        self.index_choice = []
        
        self.index_choice.append([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]])

        self.id_choice = 0  # 不同的波段选择方案

        ##


    
    def reset_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            
        if self.z_input_projection is not None:
            self.z_input_projection.apply(_init_weights)
        if self.x_input_projection is not None:
            self.x_input_projection.apply(_init_weights)

        self.decoder_f_z.apply(_init_weights)
        self.decoder_f_x.apply(_init_weights)
        self.encoder_f_z.apply(_init_weights)
        self.encoder_f_x.apply(_init_weights)

    def forward(self, search, template):
        """ The forward expects a NestedTensor, which consists of:
               - search.tensors: batched images, of shape [batch_size x 3 x H_search x W_search]
               - search.mask: a binary mask of shape [batch_size x H_search x W_search], containing 1 on padded pixels
               - template.tensors: batched images, of shape [batch_size x 3 x H_template x W_template]
               - template.mask: a binary mask of shape [batch_size x H_template x W_template], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits for all feature vectors.
                                Shape= [batch_size x num_vectors x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all feature vectors, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image.

        """
        if not isinstance(search, NestedTensor):
            # clk 22-12-19
            search_1= search[:,self.index_choice[self.id_choice][0],:,:]
            search_2= search[:,self.index_choice[self.id_choice][1],:,:]
            search_3= search[:,self.index_choice[self.id_choice][2],:,:]
            search_4= search[:,self.index_choice[self.id_choice][3],:,:]
            search_5= search[:,self.index_choice[self.id_choice][4],:,:]
            ##
            search1 = nested_tensor_from_tensor_2(search_1)
            search2 = nested_tensor_from_tensor_2(search_2)
            search3 = nested_tensor_from_tensor_2(search_3)
            search4 = nested_tensor_from_tensor_2(search_4)
            search5 = nested_tensor_from_tensor_2(search_5)
            ########## mat16 ##########

        if not isinstance(template, NestedTensor):
            # clk 22-12-19
            template_1= template[:,self.index_choice[self.id_choice][0],:,:]
            template_2= template[:,self.index_choice[self.id_choice][1],:,:]
            template_3= template[:,self.index_choice[self.id_choice][2],:,:]
            template_4= template[:,self.index_choice[self.id_choice][3],:,:]
            template_5= template[:,self.index_choice[self.id_choice][4],:,:]
            ##
            template1 = nested_tensor_from_tensor_2(template_1)
            template2 = nested_tensor_from_tensor_2(template_2)
            template3 = nested_tensor_from_tensor_2(template_3)
            template4 = nested_tensor_from_tensor_2(template_4)
            template5 = nested_tensor_from_tensor_2(template_5)
            ########## mat16 ##########
        ########## mat16 ##########
        features_search1, pos_search = self.backbone(search1)       
        features_search2, pos_search = self.backbone(search2)         
        features_search3, pos_search = self.backbone(search3)        
        features_search4, pos_search = self.backbone(search4)         
        features_search5, pos_search = self.backbone(search5)       
        feature_template1, pos_template = self.backbone(template1)        
        feature_template2, pos_template = self.backbone(template2)         
        feature_template3, pos_template = self.backbone(template3)        
        feature_template4, pos_template = self.backbone(template4)         
        feature_template5, pos_template = self.backbone(template5)         
        ########## mat16 ##########

        ########## mat16 ##########
        src_search1, mask_search= features_search1[-1].decompose()    # torch.Size([8, 1024, 32, 32])
        src_search2, mask_search= features_search2[-1].decompose()        
        src_search3, mask_search= features_search3[-1].decompose()        
        src_search4, mask_search= features_search4[-1].decompose()        
        src_search5, mask_search= features_search5[-1].decompose()        
        ########## mat16 ##########
        assert mask_search is not None          
        ########## mat16 ##########
        src_template1, mask_template= feature_template1[-1].decompose()    # torch.Size([8, 1024, 16, 16])
        src_template2, mask_template= feature_template2[-1].decompose()        
        src_template3, mask_template= feature_template3[-1].decompose()        
        src_template4, mask_template= feature_template4[-1].decompose()        
        src_template5, mask_template= feature_template5[-1].decompose() 

        ################## clk
 
        B_s,C_s,W_s,H_s = src_search1.shape
        B_z,C_z,W_z,H_z = src_template1.shape

       # decoder_feature
        # B_s,C_s,W_s,H_s = src_search1.shape
        src_search1_f = src_search1.view(B_s,C_s,W_s*H_s)  # shape:torch.Size([8, 1024, 1024])
        src_search2_f = src_search2.view(B_s,C_s,W_s*H_s)
        src_search3_f = src_search3.view(B_s,C_s,W_s*H_s)
        src_search4_f = src_search4.view(B_s,C_s,W_s*H_s)
        src_search5_f = src_search5.view(B_s,C_s,W_s*H_s)

        # B_z,C_z,W_z,H_z = src_template1.shape
        src_template1_f = src_template1.view(B_z,C_z,W_z*H_z)
        src_template2_f = src_template2.view(B_z,C_z,W_z*H_z)  # shape:torch.Size([8, 1024, 256])
        src_template3_f = src_template3.view(B_z,C_z,W_z*H_z)
        src_template4_f = src_template4.view(B_z,C_z,W_z*H_z)
        src_template5_f = src_template5.view(B_z,C_z,W_z*H_z)

        z_pos = None
        x_pos = None

        if self.z_pos_enc is not None:
            z_pos = self.z_pos_enc().unsqueeze(0)
        if self.x_pos_enc is not None:
            x_pos = self.x_pos_enc().unsqueeze(0)
        
        ########

        en_search_f1 = torch.cat([src_search1_f, src_search2_f, src_search3_f], 1)
        en_search_f2 = torch.cat([src_search4_f, src_search5_f], 1)
        en_template_f1 = torch.cat([src_template1_f, src_template2_f, src_template3_f], 1)
        en_template_f2 = torch.cat([src_template4_f, src_template5_f], 1)

        en_feat_x1, en_feat_x2 = self.encoder_f_x(en_search_f1, en_search_f2, z_pos, x_pos)
        en_feat_z1, en_feat_z2 = self.encoder_f_z(en_template_f1, en_template_f2, z_pos, x_pos)
        
        en_feat_x_1 = en_feat_x1[:, 0*1024:1*1024, :]
        en_feat_x_2 = en_feat_x1[:, 1*1024:2*1024, :]
        en_feat_x_3 = en_feat_x1[:, 2*1024:3*1024, :]
        en_feat_x_4 = en_feat_x2[:, 0*1024:1*1024, :]
        en_feat_x_5 = en_feat_x2[:, 1*1024:2*1024, :]

        en_feat_z_1 = en_feat_z1[:, 0*1024:1*1024, :]
        en_feat_z_2 = en_feat_z1[:, 1*1024:2*1024, :]
        en_feat_z_3 = en_feat_z1[:, 2*1024:3*1024, :]
        en_feat_z_4 = en_feat_z2[:, 0*1024:1*1024, :]
        en_feat_z_5 = en_feat_z2[:, 1*1024:2*1024, :]

        src_search_1_f = torch.cat([en_feat_x_1, en_feat_x_2, en_feat_x_3, en_feat_x_4, en_feat_x_5], 1)   # shape:torch.Size([8, 5120, 1024])
        src_template_1_f = torch.cat([en_feat_z_1, en_feat_z_2, en_feat_z_3, en_feat_z_4, en_feat_z_5], 1)  # shape:torch.Size([8, 1280, 1024])
        src_search_2_f = (en_feat_x_1 + en_feat_x_2 + en_feat_x_3 + en_feat_x_4 + en_feat_x_5)/5
        src_template_2_f = (en_feat_z_1 + en_feat_z_2 + en_feat_z_3 + en_feat_z_4 + en_feat_z_5)/5  

        decoder_feat_z_f = self.decoder_f_z(src_template_1_f, src_template_2_f, z_pos, x_pos)   
        src_template__f = self.out_norm_f_z(decoder_feat_z_f)  
        src_template__f = self.relu(src_template__f)    # Relu   
        src_template__f = 0.3*src_template__f + src_template_2_f  # res
        
        decoder_feat_x_f = self.decoder_f_x(src_search_1_f, src_search_2_f, z_pos, x_pos)     
        src_search__f = self.out_norm_f_x(decoder_feat_x_f)   
        src_search__f = self.relu(src_search__f)    # Relu
        src_search__f = 0.3*src_search__f + src_search_2_f    # res
        
        B,C,N = src_template__f.shape
        src_template_f = src_template__f.view(B,C,W_z,H_z)

        B,C,N = src_search__f.shape
        src_search_f = src_search__f.view(B,C,W_s,H_s)
        
        src_search = src_search_f
        src_template = src_template_f 

        # clk
        
        ########## mat16 ##########
        assert mask_template is not None
        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search, pos_template[-1], pos_search[-1])   # clk
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def track(self, search):            ###lpp###修改
        if not isinstance(search, NestedTensor):        # search.shape : torch.Size([1, 3, 256, 256])
           # clk 22-12-19
            search_1= search[:,self.index_choice[self.id_choice][0],:,:]
            search_2= search[:,self.index_choice[self.id_choice][1],:,:]
            search_3= search[:,self.index_choice[self.id_choice][2],:,:]
            search_4= search[:,self.index_choice[self.id_choice][3],:,:]
            search_5= search[:,self.index_choice[self.id_choice][4],:,:]
            ##
            search1 = nested_tensor_from_tensor_2(search_1)
            search2 = nested_tensor_from_tensor_2(search_2)
            search3 = nested_tensor_from_tensor_2(search_3)
            search4 = nested_tensor_from_tensor_2(search_4)
            search5 = nested_tensor_from_tensor_2(search_5)

        features_search1, pos_search = self.backbone(search1)         # lpp
        features_search2, pos_search = self.backbone(search2)         # lpp
        features_search3, pos_search = self.backbone(search3)         # lpp
        features_search4, pos_search = self.backbone(search4)         # lpp
        features_search5, pos_search = self.backbone(search5)         # lpp

        src_search1, mask_search = features_search1[-1].decompose()        # torch.Size([1, 1024, 32, 32]) torch.Size([1, 32, 32])
        src_search2, mask_search = features_search2[-1].decompose()        # torch.Size([1, 1024, 32, 32]) torch.Size([1, 32, 32])
        src_search3, mask_search = features_search3[-1].decompose()        # torch.Size([1, 1024, 32, 32]) torch.Size([1, 32, 32])
        src_search4, mask_search = features_search4[-1].decompose()        # torch.Size([1, 1024, 32, 32]) torch.Size([1, 32, 32])
        src_search5, mask_search = features_search5[-1].decompose()        # torch.Size([1, 1024, 32, 32]) torch.Size([1, 32, 32])

        ################## clk

        B_s,C_s,W_s,H_s = src_search1.shape

        # decoder_f

        src_search1_f = src_search1.view(B_s,C_s,W_s*H_s)
        src_search2_f = src_search2.view(B_s,C_s,W_s*H_s)
        src_search3_f = src_search3.view(B_s,C_s,W_s*H_s)
        src_search4_f = src_search4.view(B_s,C_s,W_s*H_s)
        src_search5_f = src_search5.view(B_s,C_s,W_s*H_s)

        z_pos = None
        x_pos = None

        if self.z_pos_enc is not None:
            z_pos = self.z_pos_enc().unsqueeze(0)
        if self.x_pos_enc is not None:
            x_pos = self.x_pos_enc().unsqueeze(0)

        en_search_f1 = torch.cat([src_search1_f, src_search2_f, src_search3_f], 1)
        en_search_f2 = torch.cat([src_search4_f, src_search5_f], 1)
        
        ########

        en_feat_x1, en_feat_x2 = self.encoder_f_x(en_search_f1, en_search_f2, z_pos, x_pos)
        
        en_feat_x_1 = en_feat_x1[:, 0*1024:1*1024, :]
        en_feat_x_2 = en_feat_x1[:, 1*1024:2*1024, :]
        en_feat_x_3 = en_feat_x1[:, 2*1024:3*1024, :]
        en_feat_x_4 = en_feat_x2[:, 0*1024:1*1024, :]
        en_feat_x_5 = en_feat_x2[:, 1*1024:2*1024, :]

        src_search_1_f = torch.cat([en_feat_x_1, en_feat_x_2, en_feat_x_3, en_feat_x_4, en_feat_x_5], 1)   # shape:torch.Size([8, 5120, 1024])
        src_search_2_f = (en_feat_x_1 + en_feat_x_2 + en_feat_x_3 + en_feat_x_4 + en_feat_x_5)/5
        #####

        decoder_feat_x_f = self.decoder_f_x(src_search_1_f, src_search_2_f, z_pos, x_pos)       
        src_search__f = self.out_norm_f_x(decoder_feat_x_f)   
        src_search__f = self.relu(src_search__f)    # Rel
        src_search__f = 0.3*src_search__f + src_search_2_f    # res
        
        B,C,N = src_search__f.shape
        src_search_f = src_search__f.view(B,C,W_s,H_s)

        src_search = src_search_f

        # clk

        src_template = self.zf 
        pos_template = self.pos_template
        mask_template = self.mask_template 

        # clk

        assert mask_template is not None
        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search, pos_template[-1], pos_search[-1])   # torch.Size([1, 1, 1024, 256])
        outputs_class = self.class_embed(hs)        # torch.Size([1, 1, 1024, 2])
        outputs_coord = self.bbox_embed(hs).sigmoid()       # torch.Size([1, 1, 1024, 4])
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def template(self, z):
        # clk
        template_1= z[:,self.index_choice[self.id_choice][0],:,:]
        template_2= z[:,self.index_choice[self.id_choice][1],:,:]
        template_3= z[:,self.index_choice[self.id_choice][2],:,:]
        template_4= z[:,self.index_choice[self.id_choice][3],:,:]
        template_5= z[:,self.index_choice[self.id_choice][4],:,:]
        ##
        template1 = nested_tensor_from_tensor_2(template_1)
        template2 = nested_tensor_from_tensor_2(template_2)
        template3 = nested_tensor_from_tensor_2(template_3)
        template4 = nested_tensor_from_tensor_2(template_4)
        template5 = nested_tensor_from_tensor_2(template_5)

        feature_template1, pos_template = self.backbone(template1)        
        feature_template2, pos_template = self.backbone(template2)         
        feature_template3, pos_template = self.backbone(template3)        
        feature_template4, pos_template = self.backbone(template4)         
        feature_template5, pos_template = self.backbone(template5)

        src_template1, mask_template= feature_template1[-1].decompose()    
        src_template2, mask_template= feature_template2[-1].decompose()        
        src_template3, mask_template= feature_template3[-1].decompose()        
        src_template4, mask_template= feature_template4[-1].decompose()        
        src_template5, mask_template= feature_template5[-1].decompose() 

        ################ clk 22-10-2
        B_z,C_z,W_z,H_z = src_template1.shape

        # decoder_f

        # B_z,C_z,W_z,H_z = src_template1.shape
        src_template1_f = src_template1.view(B_z,C_z,W_z*H_z)
        src_template2_f = src_template2.view(B_z,C_z,W_z*H_z)
        src_template3_f = src_template3.view(B_z,C_z,W_z*H_z)
        src_template4_f = src_template4.view(B_z,C_z,W_z*H_z)
        src_template5_f = src_template5.view(B_z,C_z,W_z*H_z)

        z_pos = None
        x_pos = None

        if self.z_pos_enc is not None:
            z_pos = self.z_pos_enc().unsqueeze(0)
        if self.x_pos_enc is not None:
            x_pos = self.x_pos_enc().unsqueeze(0)

        en_template_f1 = torch.cat([src_template1_f, src_template2_f, src_template3_f], 1)
        en_template_f2 = torch.cat([src_template4_f, src_template5_f], 1)

        ##### clk 22-10-5
        en_feat_z1, en_feat_z2 = self.encoder_f_z(en_template_f1, en_template_f2, z_pos, x_pos)
        
        en_feat_z_1 = en_feat_z1[:, 0*1024:1*1024, :]
        en_feat_z_2 = en_feat_z1[:, 1*1024:2*1024, :]
        en_feat_z_3 = en_feat_z1[:, 2*1024:3*1024, :]
        en_feat_z_4 = en_feat_z2[:, 0*1024:1*1024, :]
        en_feat_z_5 = en_feat_z2[:, 1*1024:2*1024, :]

        src_template_1_f = torch.cat([en_feat_z_1, en_feat_z_2, en_feat_z_3, en_feat_z_4, en_feat_z_5], 1)  # shape:torch.Size([8, 1280, 1024])
        src_template_2_f = (en_feat_z_1 + en_feat_z_2 + en_feat_z_3 + en_feat_z_4 + en_feat_z_5)/5 
        
        decoder_feat_z_f = self.decoder_f_z(src_template_1_f, src_template_2_f, z_pos, x_pos)       
        src_template__f = self.out_norm_f_z(decoder_feat_z_f)       
        src_template__f = self.relu(src_template__f)    # Rel 
        src_template__f = 0.3*src_template__f + src_template_2_f  # res
        
        B,C,N = src_template__f.shape
        src_template_f = src_template__f.view(B,C,W_z,H_z)

        src_template = src_template_f

        self.zf = src_template
        self.pos_template = pos_template
        self.mask_template = mask_template
        
        # clk

class SetCriterion(nn.Module):
    """ This class computes the loss for TransT.
    The process happens in two steps:
        1) we compute assignment between ground truth box and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, always be 1 for single object tracking.
            matcher: module able to compute a matching between target and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        loss_giou = 1 - giou
        iou = iou
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['iou'] = iou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the target
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)

        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos))

        return losses

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class CustomLoader(Loader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]

        super(CustomLoader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))

        if os.name == 'nt':
            filename = filename.replace('/', '\\')
        with open(filename, 'r') as f:
            return yaml.load(f, CustomLoader)
CustomLoader.add_constructor('!include', CustomLoader.include)
def load_yaml(path: str, loader=CustomLoader):
    with open(path, 'rb') as f:
        object_ = yaml.load(f, Loader=loader)
    return object_

def build_swin_track_main_components(config, num_epochs, iterations_per_epoch, event_register, has_training_run):
    transformer_config = config['transformer']

    drop_path_config = transformer_config['drop_path']
    drop_path_allocator = DropPathAllocator(drop_path_config['rate'])

    backbone_dim = transformer_config['backbone']['dim']
    transformer_dim = 1024

    # clk
    z_shape = 16
    x_shape = 32

    z_shape_1 = [z_shape*5,z_shape]
    z_shape_2 = [z_shape,z_shape]

    x_shape_1 = [x_shape*5,x_shape]
    x_shape_2 = [x_shape,x_shape]

    # clk

    backbone_out_stage = transformer_config['backbone']['stage']

    z_input_projection = None
    x_input_projection = None
    if backbone_dim != transformer_dim:
        z_input_projection = nn.Linear(backbone_dim, transformer_dim)
        x_input_projection = nn.Linear(backbone_dim, transformer_dim)

    num_heads = transformer_config['num_heads']
    mlp_ratio = transformer_config['mlp_ratio']
    qkv_bias = transformer_config['qkv_bias']
    drop_rate = transformer_config['drop_rate']
    attn_drop_rate = transformer_config['attn_drop_rate']

    position_embedding_config = transformer_config['position_embedding']
    z_pos_enc, x_pos_enc = build_position_embedding(position_embedding_config, z_shape_2, x_shape_2, transformer_dim)

    with drop_path_allocator:
        channel = 1024

        encoder_f_z = build_encoder(config, drop_path_allocator,
                                16*16, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                [1024,3], [1024,2])

        encoder_f_x = build_encoder(config, drop_path_allocator,
                                32*32, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                [1024,3], [1024,2])

        decoder_f_z = build_decoder(config, drop_path_allocator,
                                16*16, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                [1024,5], [1024,1])
        
        decoder_f_x = build_decoder(config, drop_path_allocator,
                                32*32, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                [1024,5], [1024,1])

    out_norm = nn.LayerNorm(transformer_dim)
    out_norm_f_z = nn.LayerNorm(16*16)
    out_norm_f_x = nn.LayerNorm(32*32)

    #####################

    if 'warmup' in drop_path_config and len(drop_path_allocator) > 0:
        if has_training_run:#定值
            from ltr.models_.utils.build_warmup_scheduler import build_warmup_scheduler
            scheduler = build_warmup_scheduler(drop_path_config['warmup'], drop_path_config['rate'],
                                               iterations_per_epoch, num_epochs)

            drop_path_scheduler = DropPathScheduler(drop_path_allocator.get_all_allocated(), scheduler)

            event_register.register_iteration_end_hook(drop_path_scheduler)#注释
            event_register.register_epoch_begin_hook(drop_path_scheduler)#注释
    
    ####################

    return encoder_f_z, encoder_f_x, decoder_f_z,decoder_f_x, out_norm, out_norm_f_z,out_norm_f_x,backbone_out_stage, backbone_out_stage, z_input_projection, x_input_projection, z_pos_enc, x_pos_enc
    
@model_constructor
def transt_resnet50(settings):
    # clk
    config_path = '/mnt/b2730ee6-a71b-4fb0-8470-ae63626b6f38/clk/Trans-HST/config/HST/Tiny/config.yaml'     # clk
    config = load_yaml(config_path)
    # clk
    num_classes = 1
    backbone_net = build_backbone(settings, backbone_pretrained=True)
    featurefusion_network = build_featurefusion_network(settings)
    encoder_f_z, encoder_f_x, decoder_f_z, decoder_f_x, out_norm,out_norm_f_z,out_norm_f_x, z_backbone_out_stage, x_backbone_out_stage, z_input_projection, x_input_projection, z_pos_enc, x_pos_enc = \
        build_swin_track_main_components(config, num_epochs = 300, iterations_per_epoch = None, event_register = None, has_training_run = False)

    model = TransT(
        backbone_net, featurefusion_network,  
        encoder_f_z, encoder_f_x, 
        decoder_f_z,decoder_f_x, out_norm, out_norm_f_z,out_norm_f_x,
        z_backbone_out_stage, x_backbone_out_stage,
        z_input_projection, x_input_projection,
        z_pos_enc, x_pos_enc,num_classes=num_classes
    )

    device = torch.device(settings.device)
    model.to(device)
    return model

def transt_loss(settings):
    num_classes = 1
    matcher = build_matcher()
    weight_dict = {'loss_ce': 8.334, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion
