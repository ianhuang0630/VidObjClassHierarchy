"""
Pretraining of the autoencoder for the video data to embed into a latent space
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# for feature extraction
try:
    from i3d import InceptionI3d
except:
    from src.i3d import InceptionI3d

# region proposal network
class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
    def forward(self, x):
        pass

class VanillaEnd2End(nn.Module):
    def __init__(self, frame_feat_shape, tree_embedding_dim,
                handpose_dim, handbbox_dim, timesteps, tree_level_option_nums = [20,20,20], device='cuda:0'):
        """
        Args:
            frame_feat_shape (tuple): tuple for the shape of the I3D preprocessing moudle
            tree_embedidng_dim (int): dimensionlaity of the tree embedding space trained on known classes
            handpose_dim (int): dimensionality of the handpose vector
            handbbox_dim (int): dimensionality of the hand bounding box 
        
        """
        super(VanillaEnd2End, self).__init__()

        self.frame_feat_shape = frame_feat_shape # 1024 
        self.timesteps = timesteps
        self.frame_timesteps = self.frame_feat_shape[1]

        # action_embedding weights
        self.action_embedding_dim = 3
        self.output1_dim = 5
        self.action_emb_lin1 = nn.Linear(handpose_dim+handbbox_dim, self.output1_dim)
        self.action_emb_weights1 = nn.Linear(self.output1_dim, 1, bias=False)

        self.action_emb_weights2 = nn.Linear(1024, 1, bias=False)
        self.action_emb_fc1 = nn.Linear(self.output1_dim + 1024, 128)
        self.action_emb_fc2 = nn.Linear(128, 64)
        self.action_emb_fc3 = nn.Linear(64, self.action_embedding_dim) 

        # function_encoding weights
        self.function_encoding_A = nn.Linear(self.action_embedding_dim, self.action_embedding_dim, bias=False)
        self.function_encoding_B = nn.Linear(tree_embedding_dim, tree_embedding_dim, bias=False)

        # fusion_module weights
        self.fusion_module_fc1 = nn.Linear(3 + tree_embedding_dim, 12)
        self.fusion_module_fc2 = nn.Linear(12, tree_embedding_dim)

        # tree level prediction
        embedding_dim = 8
        self.treelevel_fc1 = nn.Linear(tree_embedding_dim, embedding_dim)
        self.treelevel_fc_tl1 = nn.Linear(embedding_dim, tree_level_option_nums[0])
        self.treelevel_fc_tl2 = nn.Linear(embedding_dim + tree_level_option_nums[0],
                                tree_level_option_nums[1])
        self.treelevel_fc_tl3 = nn.Linear(embedding_dim + tree_level_option_nums[0] + tree_level_option_nums[1],
                                tree_level_option_nums[2])

        # softmax and relu
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

        self.device = device
        self.tens_timesteps = torch.Tensor([self.timesteps]).to(self.device)
        self.tens_frame_timesteps = torch.Tensor([self.frame_timesteps]).to(self.device)


    def action_embedding(self, x):
        """
        Args :
            x: dictionary with keys 'handpose', 'handbbox', 'frames', 'known'
                'frames' will be Batchsize * timesteps * D3
                'handpose' will be batchsize * timesteps * D1
                'handbbox' will be batcshize * tiemsteps * D2
                'known' will be batchsize * D_t_embedding
        Returns:
            batchsize * D4
        """


        # 1) concatenate hand pose and hand box features, and push through first linear layer
        # calculate concat * W_1^T (W_1 dim *D* x (dim handpose + dim handbbox))
        # (output dim should be TxD)
        hand_cat = torch.cat((x['handpose'], x['handbbox']), dim=2) # B x T x numfeats
        original_hand_cat_shape = hand_cat.shape
        hand_cat = hand_cat.reshape((-1, hand_cat.shape[-1]))
        hand_cat = self.action_emb_lin1(hand_cat)
        hand_cat = hand_cat.reshape((original_hand_cat_shape[0], original_hand_cat_shape[1], -1))

        # 2) derive attention weights
        # reformat the input matrix to the correct dimension before passing into affine transform
        original_hand_cat_shape = hand_cat.shape
        hand_cat = hand_cat.reshape((-1, hand_cat.shape[-1]))
        hand_attention_weights = self.softmax((self.action_emb_weights1(hand_cat))/torch.sqrt(self.tens_timesteps))
        hand_cat = hand_cat.reshape(original_hand_cat_shape)
        hand_attention_weights = hand_attention_weights.reshape((original_hand_cat_shape[0], original_hand_cat_shape[1], -1))


        # 3) aggregate according to attention
        aggregate_hand_attention = torch.matmul(hand_cat.transpose(1,2), hand_attention_weights)
        aggregate_hand_attention = aggregate_hand_attention.squeeze()

        # 4) derive attention for frames
        frames = x['frames'].transpose(1,2)
        frames_original_shape = frames.shape
        frames = frames.reshape((-1, frames_original_shape[-1]))
        frame_attention_weights = self.softmax((self.action_emb_weights2(frames))/torch.sqrt(self.tens_frame_timesteps)) # batch x timesteps x 1

        if len(frame_attention_weights.shape) == 2:
            frame_attention_weights = frame_attention_weights.unsqueeze(-1)
        frames = frames.reshape(frames_original_shape)

        # 5) aggregate according to attention
        aggregate_frame_attention = torch.matmul(frame_attention_weights, frames)
        aggregate_frame_attention = aggregate_frame_attention.squeeze()

        # 6) fully connected layers on the concatenation of aggregated hand and aggregated features
        concat_frame_hand = torch.cat((aggregate_frame_attention, aggregate_hand_attention), dim=1)
        # fully connected layers
        r = self.relu(self.action_emb_fc1(concat_frame_hand))
        r = self.relu(self.action_emb_fc2(r))
        r = self.action_emb_fc3(r)

        return r

    def function_encoding(self, r, z_u, z_k):
        """
        Args :
            x: dictionary with keys 'action_embedding', 'known', 'unknown'
                'action_embedding': batchsize * D4
                'known': batcshize * D_t_embedding
                'unknown': batchsize * D_t_embedding
        Returns:
            batchsize * D4
        """

        # 1) compute F  by F = (Ar)(Bz_u)^T 
        F = torch.matmul(self.function_encoding_A(r).unsqueeze(-1), self.function_encoding_B(z_u).unsqueeze(-1).transpose(1,2))

        # 2) output F(z_k) + r
        projected_z_u = torch.matmul(F, z_k.unsqueeze(-1)).squeeze() + r

        return projected_z_u

    def fusion_module (self, projected_z_u, z_u):
        """
        Args:
            x: dictionary with keys 'unknown_func_emb', 'unknown'
                'unknown_function_emb': batchsize * D4
                'unknown': batchsize * D_t_embedding
        Returns:
            batchsize * D_t_embedding
        """

        # 1) get delta = FC(projected_z_u, real_zu)
        z_cat = torch.cat((projected_z_u, z_u), dim=1)
        delta = self.relu(self.fusion_module_fc1(z_cat))
        delta = self.fusion_module_fc2(delta)

        # 2) output real_zu + delta
        z_u_tilde = z_u + delta

        return z_u_tilde

    def treelevel_predictor(self, z_u_tilde):
        """
        Args:
            x: dictionary with keys 'unknown_fused_emb'
        Returns:
            batchsize * layer1_choices
            batchsize * layer2_choices
            batchsize * layer3_choices
        """
        embedding = self.treelevel_fc1(z_u_tilde)

        tree_level_prediction1 = self.treelevel_fc_tl1(embedding)
        tree_level_prediction1 = self.relu(tree_level_prediction1)
        tree_level_prediction1 = self.softmax(tree_level_prediction1)

        # concatenating tree_level_predictions1
        pred1_embedding_concat = torch.cat((tree_level_prediction1, embedding), 1)
        tree_level_prediction2 = self.treelevel_fc_tl2(pred1_embedding_concat)
        tree_level_prediction2 = self.relu(tree_level_prediction2)
        tree_level_prediction2 = self.softmax(tree_level_prediction2)

        pred2_embedding_concat = torch.cat((tree_level_prediction1, tree_level_prediction2, embedding), 1)
        tree_level_prediction3 = self.treelevel_fc_tl3(pred2_embedding_concat)
        tree_level_prediction3 = self.relu(tree_level_prediction3)
        tree_level_prediction3 = self.softmax(tree_level_prediction3)

        return {'embedding': embedding,
                'tree_level_pred1': tree_level_prediction1,
                'tree_level_pred2': tree_level_prediction2,
                'tree_level_pred3': tree_level_prediction3}

    def forward(self, x):
        """
        Args:
            x: dictionary with keys 'handpose', 'handbbox', 'frames', 'unknown', 'known'
        Returns:
            batchsize * layer1_choices
            batchsize * layer2_choices
            batchsize * layer3_choices
        """

        # 1) first call attention over 'known's 
        # TODO change this to some sort of attention mechanism
        known = torch.mean(x['known'], dim=1)
        unknown = x['unknown']
        # 2) send to action embedding module
        r = self.action_embedding(x)
        project_z_u = self.function_encoding(r, unknown, known)
        z_u_tilde = self.fusion_module(project_z_u, known)

        treelevel_predictions = self.treelevel_predictor(z_u_tilde)

        return treelevel_predictions 


# action embedding module
# composed of mostly conv3ds accross the time frames
class ActionMod(nn.Module):
    def __init__(self, feature_input_shape, image_input_shape,
            num_fcs = 3, fc_output_shapes = [72, 36, 18],
            output_shape=16, time_conv_filter_width=2,
            conv_kernel_time_size = 3, conv_kernel_feat_size=3,
            feature_conv_filter_width=4, time_stride=1, feature_stride=2):

        super(ActionMod, self).__init__()
        self.output_dimension = output_dimension
        self.feature_input_shape = feature_input_shape
        self.image_input_shape = image_input_shape
        # for fully connected layers
        self.num_fcs = num_fcs
        self.fc_output_shapes = fc_output_shapes
        # for convolutions on matrix
        self.feature_stride = feature_stride
        self.time_stride = time_stride
        self.conv_kernel_time_size = conv_kernel_time_size
        self.conv_kernel_feat_size = conv_kernel_feat_size
       
        assert self.conv_kernel_time_size % 2 == 1, 'conv_kernel_time_size needs to be odd'
        assert self.conv_kernel_feat_size % 2 == 1, 'conv_kernel_feat_size needs to be odd'
        self.time_padding = int((self.conv_kernel_time_size-1)/2)
        self.feat_padding = int((self.conv_kernel_feat_size-1)/2)
        self.output_shape = output_shape # 
        
        # fully connected layer for hand/object feature input
        fc_layers = self.make_feature_fcs(self.num_fcs, self.feature_input_shape, 
                self.fc_output_shapes)
        self.feature_fc = nn.Sequential(*fc_layers)
         
        # convolutional layers for hand/object feature input
        # input should now be NxT
        conv_layers = self.make_feature_convs()
        self.feature_conv = nn.Sequential(*conv_layers)
         
        # convolution layers for image input
        image_conv_layers = self.make_image_3dconv()
        self.image_conv = nn.Sequential(*image_conv_layers)
    
    def make_feature_convs(self):
        # NOTE input would have to be (batch_size, 1, num_features, num_timesteps)
        layers = []

        layers.append(nn.Conv2d(in_channels=1, out_channels=64, 
                kernel_size=(self.conv_kernel_feat_size, self.conv_kernel_time_size), 
                stride=(self.feature_stride, self.time_stride), 
                padding=(self.feat_padding, self.time_padding)))
        layers.append(nn.ReLU())
        
        layers.append(nn.Conv2d(in_channels=64, out_channels=1,
                kernel_size=(self.conv_kernel_feat_size, self.conv_kernel_time_size), 
                stride=(self.feature_stride, self.time_stride), 
                padding=(self.feat_padding, self.time_padding)))
        layers.append(nn.ReLU()) 

        return layers

    def make_feature_fcs(self, num_layers, input_shape, output_shapes):
        layers = []
        input_shapes = [input_shape] 
        assert len(output_shapes) == num_layers

        for i in range(num_layers):
            layers.append(nn.Linear(input_shapes[-1], output_shapes[i])) # linear unit
            layers.append(nn.ReLU()) # nonlinearity
            input_shapes.append(output_shapes[i])
        
        return layers
    
    def make_image_3dconv(self):
        layers = []
        # TODO: check that the kernelsize strides along the time dimension is correct
        layers.append(nn.Conv3d(3, 64, kernel_size=(2,3,3), stride=(2,2,1), padding=(4,2,1)))
        layers.append(nn.ReLU())

        return layers

    def get_flat_fts(self, in_shape, fts):
        f = fts(Variable(torch.ones(1,*in_shape)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        frames = x['frames']
        hand_features = x['hand_features']

        feat_fc_out = self.feature_fc(precomp_feats.t()) 
        reformatted = feat_fc_out.t().view(-1, 1, feat_fc_out.shape[1], feat_fc_out.shape[0])
        feat_conv_out = self.feature_conv(reformatted)
        
        # extracting image information
        image_conv_out = self.image_conv(image)

        # TODO: final fully connected layers
        return feat_conv_out, image_conv_out

# graph embedding module
class FunctionEmbedding(nn.Module):
    def __init__(self):
        super(FunctionEmbedding, self).__init__()

    
    def foward(self, x, r):
        # r will 
        pass

# fully conne module
class FusionMod(nn.Module):
    def __init__(self):
        super(FusionMod, self).__init__()
    def forward(self, x):
        pass

# the whole damn thing
class HierarchyInsertionModel(nn.Module):
    def __init__(self):
        super(HierarchyInsertionModel, self).__init__()

    def forward(self, x):
        pass

if __name__ == '__main__':
    # features 'handpose', 'handbbox', 'frames', 'unknown', 'known'

    known = torch.randn(2, 18,10)
    unknown = torch.randn(2, 10)
    handpose = torch.randn(2, 13, 126) # 126 raw features for hand pose 
    handbbox = torch.randn(2, 13, 8) # 8 raw features for handbbox
    #frames = torch.randn(2, 13, 960, 540, 3) 
    frames = torch.randn(2, 3, 13, 224, 224)

    sample = {'unknown':unknown, 'known':known, 'handpose': handpose, 'handbbox': handbbox, 'frames':frames}

    i3d  = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load('models/i3d_pretrained/rgb_i3d_pretrained.pt'))
    i3d = i3d.to('cuda:0')

    with torch.no_grad():
        frames = sample['frames'].to('cuda:0')
        i3d_processed = i3d(frames)
        if len(i3d_processed.shape) == 2:
            i3d_processed = i3d_processed.unsqueeze(2)
        sample['frames'] = i3d_processed.to('cpu')
    import ipdb; ipdb.set_trace()
    model = VanillaEnd2End(frame_feat_shape = tuple(list(sample['frames'].shape)[1:]), tree_embedding_dim = 10,
                            handpose_dim = 126, handbbox_dim = 8, timesteps = 13, device='cpu')
    model(sample)





