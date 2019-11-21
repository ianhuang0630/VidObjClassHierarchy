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

DEVICE = 'cuda:0'

class VanillaEnd2End_Ablation(nn.Module):
    def __init__(self, frame_feat_shape, tree_embedding_dim,
                handpose_dim, handbbox_dim, timesteps, ablate = None, partial_ablation=False,
                tree_level_option_nums = [20,20,20], device='cuda:0'):
        """
        Args:
            frame_feat_shape (tuple): tuple for the shape of the I3D preprocessing moudle
            tree_embedidng_dim (int): dimensionlaity of the tree embedding space trained on known classes
            handpose_dim (int): dimensionality of the handpose vector
            handbbox_dim (int): dimensionality of the hand bounding box 
        
        """
        super(VanillaEnd2End_Ablation, self).__init__()

        # LIST OF THINGS TO ABLATE
        self.ablate=ablate
        if self.ablate is None:
            self.ablate = []
        self.partial_ablation = partial_ablation


        self.frame_feat_shape = frame_feat_shape # 1024 
        self.timesteps = timesteps
        self.frame_timesteps = self.frame_feat_shape[1]
        self.tree_embedding_dim = tree_embedding_dim

        # action_embedding weights
        self.action_embedding_dim = 3
        if 'action_embedding' in self.ablate and self.partial_ablation:
            self.action_emb_AB_map = nn.Linear(handpose_dim+handbbox_dim+1024, 3, bias=False)
            self.action_emb_out_dim = 3
        elif 'action_embedding' in self.ablate and not self.partial_ablation:
            self.action_emb_out_dim = handpose_dim+handbbox_dim+1024
        else:    
            self.output1_dim = 5
            self.action_emb_lin1 = nn.Linear(handpose_dim+handbbox_dim, self.output1_dim)
            self.action_emb_weights1 = nn.Linear(self.output1_dim, 1, bias=False)

            self.action_emb_weights2 = nn.Linear(1024, 1, bias=False)
            self.action_emb_fc1 = nn.Linear(self.output1_dim + 1024, 128)
            self.action_emb_fc2 = nn.Linear(128, 64)
            self.action_emb_fc3 = nn.Linear(64, self.action_embedding_dim) 
            self.action_emb_out_dim = self.action_embedding_dim

        # function_encoding weights
        if 'function_encoding' in self.ablate and self.partial_ablation:
            self.function_encoding_AB_map = nn.Linear(self.action_embedding_dim + self.tree_embedding_dim + self.tree_embedding_dim,
                                                    self.action_embedding_dim, bias=False)
            self.function_encoding_out_dim = self.action_embedding_dim

        elif 'function_encoding' in self.ablate and not self.partial_ablation:
            self.function_encoding_out_dim = self.action_embedding_dim + self.tree_embedding_dim + self.tree_embedding_dim

        else:
            self.function_encoding_A = nn.Linear(self.action_emb_out_dim, self.action_emb_out_dim, bias=False)
            self.function_encoding_B = nn.Linear(tree_embedding_dim, tree_embedding_dim, bias=False)
            self.function_encoding_out_dim = self.action_emb_out_dim

        if 'fusion_module' in self.ablate and self.partial_ablation:
            self.fusion_module_AB_map = nn.Linear(self.action_embedding_dim + self.tree_embedding_dim,
                                                    self.tree_embedding_dim, bias=False)
            self.fusion_module_out_dim = self.tree_embedding_dim

        elif 'fusion_module' in self.ablate and not self.partial_ablation:
            self.fusion_module_out_dim = self.action_embedding_dim + self.tree_embedding_dim

        else:
            # fusion_module weights
            self.fusion_module_fc1 = nn.Linear(self.function_encoding_out_dim + tree_embedding_dim, 12)
            self.fusion_module_fc2 = nn.Linear(12, tree_embedding_dim)
            self.fusion_module_out_dim = self.tree_embedding_dim

        # tree level prediction
        embedding_dim = 8
        self.treelevel_fc1 = nn.Linear(self.fusion_module_out_dim, embedding_dim)
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
        if 'action_embedding' in self.ablate:
            hand_cat = torch.cat((x['handpose'], x['handbbox']), dim=2)
            hand_cat = torch.mean(hand_cat, dim=1)
            frames = torch.mean(x['frames'], dim=-1)
            hcat_frames = torch.cat((hand_cat, frames), dim=1)
            if not self.partial_ablation:
                return hcat_frames
            return self.action_emb_AB_map(hcat_frames)


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
        if 'function_encoding' in self.ablate:
            concatted = torch.cat((r, z_u, z_k), dim=1)
            if not self.partial_ablation:
                return concatted
            return self.function_encoding_AB_map(concatted)

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
        if 'fusion_module' in self.ablate:
            concatted = torch.cat((projected_z_u, z_u), dim=1)
            if not self.partial_ablation:
                return concatted
            return self.fusion_module_AB_map(concatted)

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
        # data-wise ablations
        if 'known_visual' in self.ablate:
            x['known'] = torch.zeros_like(x['known'])

        if 'unknown_visual' in self.ablate:
            x['unknown'] = torch.zeros_like(x['unknown'])

        if 'video_visual' in self.ablate:
            x['frames'] = torch.zeros_like(x['frames'])

        if 'hand_pose' in self.ablate:
            x['handpose'] = torch.zeros_like(x['handpose'])

        if 'hand_bbox' in self.ablate:
            x['handbbox'] = torch.zeros_like(x['handbbox'])

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


if __name__=='__main__':
    fake_x = {}
    #hndbox
    fake_x['handbbox'] = torch.ones((16, 16, 8)).type(torch.FloatTensor).to(DEVICE)
    #hndpose
    fake_x['handpose'] = torch.ones((16, 16, 126)).type(torch.FloatTensor).to(DEVICE)
    #frames
    fake_x['frames'] = torch.ones((16, 1024, 1)).type(torch.FloatTensor).to(DEVICE)
    #unknown
    fake_x['unknown'] = torch.ones((16, 4)).type(torch.FloatTensor).to(DEVICE)
    #known
    fake_x['known'] = torch.ones((16, 1, 4)).type(torch.FloatTensor).to(DEVICE)

    mod =VanillaEnd2End_Ablation(frame_feat_shape= (1024, 1), tree_embedding_dim = 4, handpose_dim = 126,
             handbbox_dim = 8, timesteps=16, ablate=['fusion_module'], partial_ablation=False)

    mod = mod.to(DEVICE)
    out = mod(fake_x)



