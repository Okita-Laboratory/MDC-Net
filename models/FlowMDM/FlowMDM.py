import numpy as np
import torch
import torch.nn as nn
# from model.base_models import TextConditionalModel
from models.FlowMDM.x_transformers.x_transformers import ContinuousTransformerWrapper, Encoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = x.permute((1, 0, 2))
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError
class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output
class BPE_Schedule():
    def __init__(self, training_rate: float, inference_step: int, max_steps: int) -> None:
        assert training_rate >= 0 and training_rate <= 1, "training_rate must be between 0 and 1"
        assert inference_step == -1 or (
                    inference_step >= 0 and inference_step <= max_steps), "inference_step must be between 0 and max_steps"
        self.training_rate = training_rate
        self.inference_step = inference_step
        self.max_steps = max_steps
        self.last_random = None

    def step(self, t: torch.Tensor, training: bool):
        self.last_random = torch.rand(t.shape[0], device=t.device)

    def get_schedule_fn(self, t: torch.Tensor, training: bool) -> torch.Tensor:
        # False --> absolute
        # True --> relative
        if training:  # at TRAINING: then random dropout
            return self.last_random < self.training_rate
        # at INFERENCE: step function as BPE schedule
        elif self.inference_step == -1:  # --> all denoising chain with APE (absolute)
            return torch.zeros_like(t, dtype=torch.bool)
        elif self.inference_step == 0:  # --> all denoising chain with RPE (relative)
            return torch.ones_like(t, dtype=torch.bool)
        else:  # --> BPE with binary step function. Step from APE to RPE at "self.inference_step"
            return ~(t > self.max_steps - self.inference_step)

    def use_bias(self, t: torch.Tensor, training: bool) -> torch.Tensor:
        # function that returns True if we should use the absolute bias (only when using multi-segments **inference**)
        assert (t[
                    0] == t).all(), "Bias from mixed schedule only supported when using same timestep for all batch elements: " + str(
            t)
        return ~self.get_schedule_fn(t[0], training)  # if APE --> use bias to limit attention to the each subsequence

    def get_time_weights(self, t: torch.Tensor, training: bool) -> torch.Tensor:
        # 0 --> absolute
        # 1 --> relative
        return self.get_schedule_fn(t, training).to(torch.int32)
class FlowMDM(nn.Module):
    def __init__(self, njoints, nfeats, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 data_rep='rot6d', dataset='babel',
                 **kargs):
        super().__init__()
        # super().__init__(latent_dim=latent_dim, cond_mode=cond_mode, cond_mask_prob=cond_mask_prob, dropout=dropout,
        #                  clip_dim=clip_dim, clip_version=clip_version)
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.input_feats = self.njoints * self.nfeats
        self.max_seq_att = kargs.get('max_seq_att', 1024)
        self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.process_cond_input = [nn.Linear(2 * self.latent_dim, self.latent_dim) for _ in range(self.num_layers)]
        print(f"FlowMDM init")
        self.use_chunked_att = kargs.get('use_chunked_att', False)
        bpe_training_rate = kargs.get('bpe_training_ratio',
                                      0.5)  # for training, we dropout with prob 50% --> APE vs RPE
        bpe_inference_step = 125
        diffusion_steps = 1000
        self.bpe_schedule = BPE_Schedule(bpe_training_rate, bpe_inference_step, diffusion_steps)
        ws = kargs.get('rpe_horizon', -1)  # Max attention horizon
        self.local_attn_window_size = 200 if ws == -1 else ws
        print("[Training] RPE/APE rate:", bpe_training_rate)
        print(f"[Inference] BPE switch from APE to RPE at denoising step {bpe_inference_step}/{diffusion_steps}.")
        print("Local attention window size:", self.local_attn_window_size)

        self.seqTransEncoder = ContinuousTransformerWrapper(
            dim_in=self.latent_dim, dim_out=self.latent_dim,
            emb_dropout=self.dropout,
            max_seq_len=self.max_seq_att,
            use_abs_pos_emb=True,
            absolute_bpe_schedule=self.bpe_schedule,  # bpe schedule for absolute embeddings (APE)
            attn_layers=Encoder(
                dim=self.latent_dim,
                depth=self.num_layers,
                heads=self.num_heads,
                ff_mult=int(np.round(self.ff_size / self.latent_dim)),  # 2 for MDM hyper params
                layer_dropout=self.dropout, cross_attn_tokens_dropout=0,

                # ======== FLOWMDM ========
                custom_layers=('A', 'f'),  # A --> PCCAT
                custom_query_fn=None,
                # function that merges the condition into the query --> PCCAT dense layer (see Fig. 3)
                attn_max_attend_past=self.local_attn_window_size,
                attn_max_attend_future=self.local_attn_window_size,
                # ======== RELATIVE POSITIONAL EMBEDDINGS ========
                rotary_pos_emb=True,  # rotary embeddings
                rotary_bpe_schedule=self.bpe_schedule,  # bpe schedule for rotary embeddings (RPE)
            )
        )

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)
        self.num_frames = 20
        self.time_embed_dim = 512
        self.cond_embed = nn.Linear(self.input_feats * self.num_frames, self.time_embed_dim)

    def forward(self, x, timesteps, mod=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        inside y: model_kwargs with mask, pe_bias, pos_pe_abs, conditions_mask. See DiffusionWrapper_FlowMDM.
        """
        # bs, njoints, nfeats, nframes = x.shape
        bs, nframes, nfeat = x.shape
        #["y"]["mask"] = th.ones((bs, nframes), device=device, dtype=th.bool)

        y = {"mask": torch.ones((bs, nframes), device=t.device, dtype=torch.bool)}
        mask = (y['mask'].reshape((bs, nframes))[:, :nframes].to(x.device)).bool()  # [bs, max_frames]

        self.bpe_schedule.step(timesteps,
                               self.training)  # update the BPE scheduler (decides either APE or RPE for each timestep)
        if self.training or self.bpe_schedule.use_bias(timesteps, self.training):
            pe_bias = y.get("pe_bias",
                            None)  # This is for limiting the attention to inside each conditioned subsequence. The BPE will decide if we use it or not depending on the dropout at training time.
            chunked_attn = False
        else:  # when using RPE at inference --> we use the bias to limit the attention to the each subsequence
            pe_bias = None
            chunked_attn = self.use_chunked_att  # faster attention for inference with RPE for very long sequences (see LongFormer paper for details)

        # store info needed for the relative PE --> rotary embedding
        rotary_kwargs = {'timesteps': timesteps, 'pos_pe_abs': y.get("pos_pe_abs", None), 'training': self.training,
                         'pe_bias': pe_bias}

        # ============== INPUT PROCESSING ==============
        emb = self.embed_timestep(timesteps)
        if mod is not None:
            mod_proj = self.cond_embed(mod.reshape(bs, -1))
            emb = emb + mod_proj
        x = self.input_process(x)  # [seqlen, bs, d]

        # ============== MAIN ARCHITECTURE ==============
        # APE or RPE is injected inside seqTransEncoder forward function
        x, emb = x.permute(1, 0, 2), emb.permute(1, 0, 2)
        output = self.seqTransEncoder(x, mask=mask, cond_tokens=emb, attn_bias=pe_bias, rotary_kwargs=rotary_kwargs,
                                      chunked_attn=chunked_attn)  # [bs, seqlen, d]
        output = output.permute(1, 0, 2)  # [seqlen, bs, d]

        output = self.output_process(output)
        output = output.permute((0, 3, 1, 2)).view(bs, 20, 48)

        # ============== OUTPUT PROCESSING ==============
        return output

if __name__=='__main__':
    batch_size = 64
    Model = FlowMDM(
        njoints=16,
        nfeats=3,
        translation=True,
        pose_rep='xyz',
        glob=True,
        glob_rot=True,
    )
    x = torch.randn((batch_size, 20, 48))
    t = torch.randint(1000, (batch_size, ))
    y = Model(x, t)
    print(y.shape)