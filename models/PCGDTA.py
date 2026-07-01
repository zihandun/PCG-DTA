import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, GlobalAttention

class PCGDTA(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=54, output_dim=128, dropout_rate=0.2):
        super(PCGDTA, self).__init__()

        self.output_dim = output_dim
        self.n_filters = n_filters
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        
        # --- GIN MLP 构造器 ---
        def make_gin_mlp(input_dim, output_dim):
            return nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU()
            )

        # =====================================================================
        # 1. Drug Graph Layers (保留)
        # =====================================================================
        self.d_gin1 = GINConv(make_gin_mlp(num_features_xd, embed_dim))
        self.d_gin2 = GINConv(make_gin_mlp(embed_dim, embed_dim))
        self.d_gin3 = GINConv(make_gin_mlp(embed_dim, embed_dim))

        self.fc_druggraph = nn.Linear(embed_dim, output_dim)
        self.bn_fc_druggraph = nn.BatchNorm1d(output_dim)

        self.d_gate_nn = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.Tanh(), nn.Linear(embed_dim // 2, 1))
        self.d_pool_att = GlobalAttention(gate_nn=self.d_gate_nn)
        self.d_res_scale = nn.Parameter(torch.tensor(0.1))

        # =====================================================================
        # 2. Target Graph Layers (保留)
        # =====================================================================
        self.num_features_xt_graph = num_features_xt
        self.t_gin1 = GINConv(make_gin_mlp(self.num_features_xt_graph, embed_dim))
        self.t_gin2 = GINConv(make_gin_mlp(embed_dim, embed_dim))
        self.t_gin3 = GINConv(make_gin_mlp(embed_dim, embed_dim))

        self.fc_targetgraph = nn.Linear(embed_dim, output_dim)
        self.bn_fc_targetgraph = nn.BatchNorm1d(output_dim)

        self.t_gate_nn = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.Tanh(), nn.Linear(embed_dim // 2, 1))
        self.t_pool_att = GlobalAttention(gate_nn=self.t_gate_nn)
        self.t_res_scale = nn.Parameter(torch.tensor(0.1))

        # =====================================================================
        # 3. Drug Sequence (保留纯卷�?
        # =====================================================================
        self.embedding_xd = nn.Embedding(100, embed_dim)
        self.conv_xd_1 = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        self.bn_xd_1 = nn.BatchNorm1d(n_filters)
        self.conv_xd_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=3, padding=1)
        self.bn_xd_2 = nn.BatchNorm1d(n_filters*2)
        self.conv_xd_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*3, kernel_size=3, padding=1)
        self.bn_xd_3 = nn.BatchNorm1d(n_filters*3)
        self.fc_xd_seq = nn.Linear(n_filters*3, output_dim)
        self.bn_xd_seq = nn.BatchNorm1d(output_dim)

        # =====================================================================
        # 4. Target Sequence (保留纯卷�?
        # =====================================================================
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        self.bn_xt_1 = nn.BatchNorm1d(n_filters)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=3, padding=1)
        self.bn_xt_2 = nn.BatchNorm1d(n_filters*2)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*3, kernel_size=3, padding=1)
        self.bn_xt_3 = nn.BatchNorm1d(n_filters*3)
        self.fc_xt_seq = nn.Linear(n_filters*3, output_dim)
        self.bn_xt_seq = nn.BatchNorm1d(output_dim)

        # =====================================================================
        # 5. Projection Heads & Strategy C (修改重点)
        # =====================================================================
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # [修改] 删除了单独的 graph/seq 投影头，只保留用于交互对比的投影�?
        self.proj_dim = 128
        self.fused_dim = output_dim * 2 # 128 + 128 = 256
        
        # 用于计算 CL Loss �?生成门控分数的投影层
        self.inter_drug_proj = nn.Sequential(
            nn.Linear(self.fused_dim, output_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate), 
            nn.Linear(output_dim, self.proj_dim)
        )
        self.inter_target_proj = nn.Sequential(
            nn.Linear(self.fused_dim, output_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate), 
            nn.Linear(output_dim, self.proj_dim)
        )

        # [新增] 策略 C: 虚拟漏斗的参�?
        # gate_alpha: 门控的缩放系数，设为可学习，让模型自己决定抑制的力度
        self.gate_alpha = nn.Parameter(torch.tensor(1.0)) 
        # gate_beta: 残差保底�?(Residual Floor)，防止硬拒绝 (Hard Rejection)
        self.gate_beta = 0.5   # KIBA : 0.2 Davis: 0.5

        # 最终预�?MLP
        self.fc_concat1 = nn.Linear(self.fused_dim * 2, 512)
        self.bn_fc_concat1 = nn.BatchNorm1d(512)
        self.fc_concat2 = nn.Linear(512, 256)
        self.bn_fc_concat2 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, n_output)

    # -------------------------------------------------------------------------
    # Helper Functions (保留)
    # -------------------------------------------------------------------------
    def add_global_context_node(self, x, edge_index, batch, context_features):
        device = x.device
        num_curr_nodes = x.size(0)
        batch_size = context_features.size(0)
        x_new = torch.cat([x, context_features], dim=0)
        new_node_batch_indices = torch.arange(batch_size, device=device)
        batch_new = torch.cat([batch, new_node_batch_indices], dim=0)
        context_node_global_indices = torch.arange(batch_size, device=device) + num_curr_nodes
        src = torch.arange(num_curr_nodes, device=device)
        dst = context_node_global_indices[batch]
        edge_fwd = torch.stack([src, dst], dim=0)
        edge_bwd = torch.stack([dst, src], dim=0)
        edge_index_new = torch.cat([edge_index, edge_fwd, edge_bwd], dim=1)
        return x_new, edge_index_new, batch_new

    def compute_residual_pool(self, x, batch, att_layer, scale_param):
        g_mean = global_mean_pool(x, batch)
        g_att = att_layer(x, batch)
        g_final = g_mean + scale_param * g_att
        return g_final

    # -------------------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------------------
    def forward(self, DrugData, TargetData):
        x_d, edge_index_d, batch_d = DrugData.x, DrugData.edge_index, DrugData.batch
        smiles = DrugData.smiles
        x_t, edge_index_t, batch_t = TargetData.x, TargetData.edge_index, TargetData.batch
        target = TargetData.target

        # --- Phase 1: GIN (Layer 1&2) ---
        x_d = self.d_gin1(x_d, edge_index_d)
        x_t = self.t_gin1(x_t, edge_index_t)
        x_d = self.d_gin2(x_d, edge_index_d)
        x_t = self.t_gin2(x_t, edge_index_t)

        # --- Phase 2: Context Injection ---
        global_d_l2 = self.compute_residual_pool(x_d, batch_d, self.d_pool_att, self.d_res_scale)
        global_t_l2 = self.compute_residual_pool(x_t, batch_t, self.t_pool_att, self.t_res_scale)

        x_d_new, edge_index_d_new, batch_d_new = self.add_global_context_node(x_d, edge_index_d, batch_d, global_t_l2)
        x_t_new, edge_index_t_new, batch_t_new = self.add_global_context_node(x_t, edge_index_t, batch_t, global_d_l2)

        # --- Phase 3: GIN (Layer 3) ---
        x_d = self.d_gin3(x_d_new, edge_index_d_new)
        x_t = self.t_gin3(x_t_new, edge_index_t_new)

        # --- Phase 4: Graph Readout ---
        graph_xd = self.compute_residual_pool(x_d, batch_d_new, self.d_pool_att, self.d_res_scale)
        graph_xt = self.compute_residual_pool(x_t, batch_t_new, self.t_pool_att, self.t_res_scale)

        graph_xd = self.dropout(self.relu(self.bn_fc_druggraph(self.fc_druggraph(graph_xd))))
        graph_xt = self.dropout(self.relu(self.bn_fc_targetgraph(self.fc_targetgraph(graph_xt))))

        # ============================================================
        # Drug Seq Forward
        # ============================================================
        embedded_xd = self.embedding_xd(smiles).permute(0, 2, 1)
        conv_xd = self.relu(self.bn_xd_1(self.conv_xd_1(embedded_xd)))
        conv_xd = self.relu(self.bn_xd_2(self.conv_xd_2(conv_xd)))
        conv_xd = self.relu(self.bn_xd_3(self.conv_xd_3(conv_xd)))
        conv_xd = F.adaptive_max_pool1d(conv_xd, output_size=1).view(conv_xd.size(0), -1)
        conv_xd = self.dropout(self.bn_xd_seq(self.relu(self.fc_xd_seq(conv_xd))))

        # ============================================================
        # Target Seq Forward
        # ============================================================
        embedded_xt = self.embedding_xt(target).permute(0, 2, 1)
        conv_xt = self.relu(self.bn_xt_1(self.conv_xt_1(embedded_xt)))
        conv_xt = self.relu(self.bn_xt_2(self.conv_xt_2(conv_xt)))
        conv_xt = self.relu(self.bn_xt_3(self.conv_xt_3(conv_xt)))
        conv_xt = F.adaptive_max_pool1d(conv_xt, output_size=1).view(conv_xt.size(0), -1)
        conv_xt = self.dropout(self.bn_xt_seq(self.relu(self.fc_xt_seq(conv_xt))))

        # ============================================================
        # Fusion & Strategy C (Virtual Funnel Gating)
        # ============================================================
        # 1. 拼接 Graph �?Seq 特征
        drug_final = torch.cat([graph_xd, conv_xd], dim=1)   # [Batch, 256]
        target_final = torch.cat([graph_xt, conv_xt], dim=1) # [Batch, 256]

        # 2. 投影到对比空�?(Global Interaction Space)
        # 这里使用 Normalize 很重要，因为我们要算 Cosine Similarity
        z_drug = F.normalize(self.inter_drug_proj(drug_final), dim=1)
        z_target = F.normalize(self.inter_target_proj(target_final), dim=1)

        # 3. 计算对齐分数 (Cosine Similarity)
        # dim=1 是特征维度，结果 shape: [Batch, 1]
        sim_score = torch.sum(z_drug * z_target, dim=1, keepdim=True)

        # 4. 生成软门�?(Soft Gate with Residual Floor)
        # 策略 C 核心公式: Gate = alpha * Sigmoid(Sim) + beta
        # Sim 范围 -1~1, Sigmoid �?0.26~0.73 (假设 alpha=1, beta=0.2)
        # 如果匹配度高，Sim->1, Gate->接近1.2 (增强)
        # 如果不匹配，Sim->-1, Gate->0.46 (抑制，但不为0)
        gate = self.gate_alpha * torch.sigmoid(sim_score) + self.gate_beta

        # 5. 执行早期抑制 (Early Suppression)
        xc = torch.cat([drug_final, target_final], dim=1) # [Batch, 512]
        
        # 将门控应用到回归特征�?(广播乘法)
        # 物理意义：如�?CL 认为不匹配，这里的特征幅度会被整体压低，减少噪声
        xc_gated = xc * gate 

        # 6. 回归预测
        out_feat = self.dropout(self.bn_fc_concat1(self.relu(self.fc_concat1(xc_gated))))
        out_feat = self.dropout(self.bn_fc_concat2(self.relu(self.fc_concat2(out_feat))))
        out = self.out(out_feat)

        # 返回 z_drug �?z_target 用于外部计算 Contrastive Loss
        return out, z_drug, z_target