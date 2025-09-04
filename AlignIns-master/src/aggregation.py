import copy
import torch
from torch.nn.utils import parameters_to_vector
import numpy as np
import logging
from utils import vector_to_model, vector_to_name_param

import sklearn.metrics.pairwise as smp
from geom_median.torch import compute_geometric_median

# --- Start of HyDRA specific imports ---
import math
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable


# --- End of HyDRA specific imports ---


# --- Helper code for HyDRA from Snowball ---

# VAE Model and training utilities
class MyDST(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class VAE(nn.Module):
    def __init__(self, input_dim=3, latent_dim=2, hidden_dim=16):
        super(VAE, self).__init__()
        self.fc_e1 = nn.Linear(input_dim, hidden_dim)
        self.fc_e2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_d1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_d2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_d3 = nn.Linear(hidden_dim, input_dim)
        self.input_dim = input_dim

    def encoder(self, x_in):
        x = F.relu(self.fc_e1(x_in.view(-1, self.input_dim)))
        x = F.relu(self.fc_e2(x))
        mean = self.fc_mean(x)
        logvar = F.softplus(self.fc_logvar(x))
        return mean, logvar

    def decoder(self, z):
        z = F.relu(self.fc_d1(z))
        z = F.relu(self.fc_d2(z))
        x_out = self.fc_d3(z)  # Using linear output for non-image features
        return x_out.view(-1, self.input_dim)

    def sample_normal(self, mean, logvar):
        sd = torch.exp(logvar * 0.5)
        e = Variable(torch.randn_like(sd))
        z = e.mul(sd).add_(mean)
        return z

    def forward(self, x_in):
        z_mean, z_logvar = self.encoder(x_in)
        z = self.sample_normal(z_mean, z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mean, z_logvar


def train_vae(vae, data, num_epoch, device):
    if not data:
        logging.warning("VAE training data is empty. Skipping training.")
        return VAE().to(device) if vae is None else vae.to(device)

    kl_loss_fn = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    recon_loss_fn = torch.nn.MSELoss(reduction='sum')

    data = torch.tensor(data, dtype=torch.float32).to(device)

    if vae is None:
        vae = VAE().to(device)

    vae.train()
    # Ensure batch size is not larger than the number of samples
    batch_size = min(8, len(data))
    train_loader = DataLoader(MyDST(data), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(vae.parameters())

    for epoch in range(num_epoch):
        for _, x in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            recon_x, mu, logvar = vae(x)

            recon = recon_loss_fn(recon_x, x)
            kl = kl_loss_fn(mu, logvar)

            loss = recon + kl
            loss.backward()
            optimizer.step()
    return vae


# --- End of Helper code for HyDRA ---


class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = args.server_lr
        self.n_params = n_params
        # Persist previous global params to compute temporal direction (TDA) properly
        self.prev_global_params = None

        if self.args.aggr == 'foolsgold':
            self.memory_dict = dict()
            self.wv_history = []

    def aggregate_updates(self, global_model, agent_updates_dict, reputation_scores):

        lr_vector = torch.Tensor([self.server_lr] * self.n_params).to(self.args.device)
        if self.args.aggr != "rlr":
            lr_vector = lr_vector
        else:
            lr_vector, _ = self.compute_robustLR(agent_updates_dict)

        aggregated_updates = 0
        cur_global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()

        if self.args.aggr == 'avg' or self.args.aggr == 'rlr' or self.args.aggr == 'lockdown':
            aggregated_updates = self.agg_avg(agent_updates_dict)
        elif self.args.aggr == 'hydra':
            aggregated_updates, reputation_scores = self.agg_hydra(agent_updates_dict, global_model, reputation_scores)
        elif self.args.aggr == 'alignins':
            aggregated_updates = self.agg_alignins(agent_updates_dict, cur_global_params)
        elif self.args.aggr == 'mmetric':
            aggregated_updates = self.agg_mul_metric(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'foolsgold':
            aggregated_updates = self.agg_foolsgold(agent_updates_dict)
        elif self.args.aggr == "mkrum":
            aggregated_updates = self.agg_mkrum(agent_updates_dict)
        elif self.args.aggr == "rfa":
            aggregated_updates = self.agg_rfa(agent_updates_dict)

        neurotoxin_mask = {}
        if torch.is_tensor(aggregated_updates) and aggregated_updates.nelement() > 0:
            updates_dict = vector_to_name_param(aggregated_updates, copy.deepcopy(global_model.state_dict()))
            for name in updates_dict:
                updates = updates_dict[name].abs().view(-1)
                gradients_length = torch.numel(updates)
                if gradients_length > 0:
                    _, indices = torch.topk(-1 * updates, int(gradients_length * self.args.dense_ratio))
                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1
                    neurotoxin_mask[name] = (mask_flat.reshape(updates_dict[name].size()))

        cur_global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        new_global_params = (cur_global_params + lr_vector * aggregated_updates).float()
        vector_to_model(new_global_params, global_model)
        # Update temporal reference after applying this round
        self.prev_global_params = cur_global_params.clone().detach()
        return updates_dict, neurotoxin_mask, reputation_scores

    def agg_hydra(self, agent_updates_dict, global_model, reputation_scores):
        logging.info("--- Starting HyDRA Aggregation (Optimized with MPSA features) ---")

        # --- Step 1: Robust Bottom-up Election (from Snowball) ---
        agent_updates_list = list(agent_updates_dict.values())
        K = len(agent_updates_list)
        agent_ids = list(agent_updates_dict.keys())
        model_template = global_model.state_dict()
        agent_updates_as_dicts = [vector_to_name_param(update_vec.clone(), copy.deepcopy(model_template)) for update_vec
                                  in agent_updates_list]
        model_keys = list(model_template.keys())
        key_layers = [key for key in model_keys if ('conv' in key or 'classifier' in key) and 'weight' in key]
        logging.info(f"Performing election on key layers: {key_layers}")

        total_votes = np.zeros(K)
        num_clusters = self.args.ct + 1
        for layer_name in key_layers:
            try:
                layer_updates = [update_dict[layer_name].view(-1).cpu().numpy() for update_dict in
                                 agent_updates_as_dicts]
                layer_updates_numpy = np.array(layer_updates)
            except KeyError:
                logging.warning(f"Layer {layer_name} not found in updates, skipping.")
                continue

            benign_lists_per_layer = []
            scores_per_layer = []
            for i in range(K):
                try:
                    # =================================================================
                    # CORE FIX: Create difference vectors from each client's perspective
                    # =================================================================
                    diff_from_i = layer_updates_numpy - layer_updates_numpy[i]

                    distances = np.linalg.norm(diff_from_i, axis=1)
                    farthest_indices = np.argsort(-distances)[:self.args.ct]

                    # Centroids are also from the difference space
                    initial_centroid_indices = [i] + farthest_indices.tolist()
                    initial_centroids = diff_from_i[initial_centroid_indices]

                    kmeans = KMeans(n_clusters=num_clusters, init=initial_centroids, n_init=1, random_state=0).fit(
                        diff_from_i)  # Cluster on difference vectors

                    cluster_labels = kmeans.labels_
                    my_cluster = cluster_labels[i]
                    voted_indices = np.where(cluster_labels == my_cluster)[0]

                    if len(np.unique(cluster_labels)) > 1:
                        # Score is also calculated on the difference space
                        score = calinski_harabasz_score(diff_from_i, cluster_labels)
                    else:
                        score = 0
                    benign_lists_per_layer.append(voted_indices)
                    scores_per_layer.append(score)
                except Exception as e:
                    logging.warning(f"Clustering for client {i} on layer {layer_name} failed: {e}")
                    benign_lists_per_layer.append([])
                    scores_per_layer.append(0)

            scores_per_layer = np.array(scores_per_layer)
            effective_ids = np.where(scores_per_layer > 0)[0]
            if len(effective_ids) > 0:
                effective_scores = scores_per_layer[effective_ids]
                if np.max(effective_scores) > np.min(effective_scores):
                    normalized_scores = (effective_scores - np.min(effective_scores)) / (
                            np.max(effective_scores) - np.min(effective_scores))
                else:
                    normalized_scores = np.ones_like(effective_scores)
                for voter_idx, normalized_score in zip(effective_ids, normalized_scores):
                    for voted_idx in benign_lists_per_layer[voter_idx]:
                        total_votes[voted_idx] += normalized_score

        if np.max(total_votes) > 0:
            vote_scores = total_votes / np.max(total_votes)
        else:
            vote_scores = np.zeros_like(total_votes)
        M_tilde = max(2, int(0.25 * K))
        trusted_seed_indices = np.argsort(-total_votes)[:M_tilde].tolist()
        logging.info(f"Trusted seed indices: {[agent_ids[i] for i in trusted_seed_indices]}")
        malicious_seeds = [agent_ids[i] for i in trusted_seed_indices if agent_ids[i] < self.args.num_corrupt]
        logging.info(
            f"Malicious clients in trusted seeds: {len(malicious_seeds)}/{len(trusted_seed_indices)} -> {malicious_seeds}")

        # --- Step 2: Enhanced Feature Engineering for VAE ---
        if not trusted_seed_indices:
            logging.warning("No trusted seeds found. Returning zero update.")
            return torch.zeros_like(agent_updates_list[0]), reputation_scores

        # Compute Temporal Direction Alignment (TDA) against prior global update direction
        tda_scores = []
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        # Prefer temporal direction: theta_t-1 - theta_t-2. If unavailable, use mean of trusted seeds' updates later
        if hasattr(self, 'prev_global_params') and self.prev_global_params is not None:
            current_flat = parameters_to_vector([p.data for p in global_model.parameters()]).detach()
            global_direction = (current_flat - self.prev_global_params)
        else:
            global_direction = None
        if global_direction is not None and torch.norm(global_direction) > 1e-12:
            for update in agent_updates_list:
                tda_scores.append(cos(update, global_direction).item())
        else:
            # Fallback: use average of all updates as proxy direction
            proxy_direction = torch.mean(torch.stack(agent_updates_list, dim=0), dim=0)
            for update in agent_updates_list:
                tda_scores.append(cos(update, proxy_direction).item())
        logging.info('TDA Scores: %s' % [round(s, 4) for s in tda_scores])

        mpsa_scores = []
        inter_model_updates = torch.stack(agent_updates_list, dim=0)
        # Compute major sign using only trusted seeds to reduce attacker influence
        seed_updates = inter_model_updates[trusted_seed_indices] if len(trusted_seed_indices) > 0 else inter_model_updates
        major_sign = torch.sign(torch.sum(torch.sign(seed_updates), dim=0))
        for update in agent_updates_list:
            k = max(1, int(len(update) * self.args.sparsity))
            _, topk_indices = torch.topk(torch.abs(update), k)
            agreement = torch.sum(torch.sign(update[topk_indices]) == major_sign[topk_indices])
            score = (agreement.float() / float(k)).item()
            mpsa_scores.append(score)
        logging.info('MPSA Scores: %s' % [round(s, 4) for s in mpsa_scores])

        tda_scores_np = np.array(tda_scores)
        mpsa_scores_np = np.array(mpsa_scores)
        vote_scores_np = np.array(vote_scores)

        if np.std(tda_scores_np) > 1e-9:
            tda_scores_normalized = (tda_scores_np - np.mean(tda_scores_np)) / np.std(tda_scores_np)
        else:
            tda_scores_normalized = np.zeros_like(tda_scores_np)

        if np.std(mpsa_scores_np) > 1e-9:
            mpsa_scores_normalized = (mpsa_scores_np - np.mean(mpsa_scores_np)) / np.std(mpsa_scores_np)
        else:
            mpsa_scores_normalized = np.zeros_like(mpsa_scores_np)

        if np.std(vote_scores_np) > 1e-9:
            vote_scores_normalized = (vote_scores_np - np.mean(vote_scores_np)) / np.std(vote_scores_np)
        else:
            vote_scores_normalized = np.zeros_like(vote_scores_np)

        behavioral_features = np.array(
            [[mpsa_scores_normalized[i], tda_scores_normalized[i], vote_scores_normalized[i]] for i in
             range(K)])

        # --- Step 3: VAE-based Anomaly Detection ---
        trusted_features_np = behavioral_features[trusted_seed_indices]

        # =================================================================
        # OPTIMIZATION: Robust VAE training by filtering seed outliers
        # =================================================================
        if len(trusted_features_np) > 1:
            medians = np.median(trusted_features_np, axis=0)
            mads = np.median(np.abs(trusted_features_np - medians), axis=0)

            # To avoid division by zero if a feature is constant
            mads[mads < 1e-9] = 1e-9

            robust_z_scores = np.abs(trusted_features_np - medians) / mads

            # Filter out seeds where any feature has a z-score > 2.5
            outlier_mask = np.any(robust_z_scores > 2.5, axis=1)

            filtered_seed_indices = [idx for i, idx in enumerate(trusted_seed_indices) if not outlier_mask[i]]

            if len(filtered_seed_indices) < len(trusted_seed_indices):
                logging.info(
                    f"Filtered {len(trusted_seed_indices) - len(filtered_seed_indices)} outlier seeds before VAE training.")

            if not filtered_seed_indices:
                logging.warning("All trusted seeds were outliers. Using original seeds.")
                filtered_seed_indices = trusted_seed_indices

            trusted_features = behavioral_features[filtered_seed_indices].tolist()
        else:
            trusted_features = trusted_features_np.tolist()

        logging.info(f"Training VAE on pure features from {len(trusted_features)} trusted seeds.")

        vae = train_vae(None, trusted_features, num_epoch=50, device=self.args.device)
        vae.eval()

        anomaly_scores = []
        behavioral_features_tensor = torch.tensor(behavioral_features, dtype=torch.float32).to(self.args.device)
        with torch.no_grad():
            reconstructed_features, _, _ = vae(behavioral_features_tensor)
            for i in range(K):
                recon_error = torch.norm(behavioral_features_tensor[i] - reconstructed_features[i], p=2).item()
                anomaly_scores.append(recon_error)
        logging.info('Anomaly Scores (Reconstruction Errors): %s' % [round(s, 4) for s in anomaly_scores])

        # --- Step 4: Final Selection with Adaptive Thresholding ---
        anomaly_scores_np = np.array(anomaly_scores)
        alpha = 0.5
        final_scores = (1 - alpha) * anomaly_scores_np - alpha * np.array(reputation_scores)

        # =================================================================
        # NEW LOGGING: Log the final combined scores
        # =================================================================
        logging.info('Final Combined Scores: %s' % [round(s, 4) for s in final_scores])

        median_score = np.median(final_scores)
        mad_score = np.median(np.abs(final_scores - median_score))
        if mad_score < 1e-9: mad_score = 1.4826 * np.std(final_scores)  # Fallback to std dev if MAD is zero

        score_threshold = median_score + 2.5 * mad_score
        benign_indices = np.where(final_scores < score_threshold)[0].tolist()

        min_clients = max(M_tilde, int(0.3 * K))
        max_clients = self.args.num_agents - self.args.num_corrupt

        if len(benign_indices) < min_clients:
            logging.warning(
                f"Adaptive threshold resulted in only {len(benign_indices)} clients. Expanding to the minimum of {min_clients}.")
            benign_indices = np.argsort(final_scores)[:min_clients].tolist()
        elif len(benign_indices) > max_clients:
            logging.warning(
                f"Adaptive threshold resulted in {len(benign_indices)} clients. Capping at the maximum of {max_clients}.")
            benign_indices = np.argsort(final_scores)[:max_clients].tolist()

        # =================================================================
        # NEW LOGGING: Log the final selected client IDs
        # =================================================================
        benign_ids_selected = [agent_ids[i] for i in benign_indices]
        malicious_final = [i for i in benign_ids_selected if i < self.args.num_corrupt]
        logging.info(f"Final selected benign IDs: {benign_ids_selected}")
        logging.info(
            f"Malicious clients in final selection: {len(malicious_final)}/{len(benign_indices)} -> {malicious_final}")

        # --- Step 5: Update Reputation and Aggregate ---
        if np.max(anomaly_scores_np) > np.min(anomaly_scores_np):
            norm_anomaly = (anomaly_scores_np - np.min(anomaly_scores_np)) / (
                    np.max(anomaly_scores_np) - np.min(anomaly_scores_np))
        else:
            norm_anomaly = np.zeros_like(anomaly_scores_np)

        beta = 0.1
        # =================================================================
        # OPTIMIZATION: Update reputation score based on median
        # =================================================================
        median_norm_anomaly = np.median(norm_anomaly)
        for i in range(len(reputation_scores)):
            reputation_scores[i] -= beta * (norm_anomaly[i] - median_norm_anomaly)

        logging.info('Updated Reputation Scores: %s' % [round(s, 4) for s in reputation_scores])

        if not benign_indices:
            logging.warning("No benign clients selected after VAE filtering. Returning zero update.")
            return torch.zeros_like(agent_updates_list[0]), reputation_scores

        benign_updates_list = [agent_updates_list[i] for i in benign_indices]
        benign_updates_tensor = torch.stack(benign_updates_list, dim=0)
        updates_norm = torch.norm(benign_updates_tensor, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()

        clipped_updates_dict = {}
        for i, original_idx in enumerate(benign_indices):
            client_id = agent_ids[original_idx]
            update = benign_updates_list[i]
            norm = updates_norm[i].item()
            clipped_update = update * min(1.0, norm_clip / norm) if norm > 1e-6 else update
            clipped_updates_dict[client_id] = clipped_update

        aggregated_update = self.agg_avg(clipped_updates_dict)
        logging.info("--- HyDRA Aggregation Finished ---")
        return aggregated_update, reputation_scores

    def agg_rfa(self, agent_updates_dict):
        local_updates = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)

        n = len(local_updates)
        temp_updates = torch.stack(local_updates, dim=0)
        weights = torch.ones(n).to(self.args.device)
        gw = compute_geometric_median(temp_updates, weights).median
        for i in range(2):
            weights = torch.mul(weights, torch.exp(-1.0 * torch.norm(temp_updates - gw, dim=1)))
            gw = compute_geometric_median(temp_updates, weights).median

        aggregated_model = gw
        return aggregated_model

    def agg_alignins(self, agent_updates_dict, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = list(agent_updates_dict.keys())
        num_chosen_clients = len(chosen_clients)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]),
                                         int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(
                torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(
                inter_model_updates[i][init_indices])).item())

            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())

        logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])

        ######## MZ-score calculation ########
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)

        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std if mpsa_std > 1e-9 else 0)

        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])

        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std if tda_std > 1e-9 else 0)

        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

        ######## Anomaly detection with MZ score ########

        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(
            set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(
            set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

        benign_set = benign_idx2.intersection(benign_idx1)

        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates_tensor = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        ######## Post-filtering model clipping ########

        updates_norm = torch.norm(benign_updates_tensor, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()

        clipped_updates_dict = {}

        for i, idx in enumerate(benign_idx):
            client_id = chosen_clients[idx]
            update = local_updates[idx]
            norm = updates_norm[i]
            clipped_update = update * min(1.0, norm_clip / norm.item())
            clipped_updates_dict[client_id] = clipped_update

        logging.info('selected update index: %s' % str([chosen_clients[i] for i in benign_idx]))

        aggregated_update = self.agg_avg(clipped_updates_dict)
        return aggregated_update

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """
        if not agent_updates_dict:
            return torch.zeros(self.n_params, device=self.args.device)

        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates += n_agent_data * update
            total_data += n_agent_data

        if total_data == 0:
            return torch.zeros(self.n_params, device=self.args.device)

        return sm_updates / total_data

    def agg_mkrum(self, agent_updates_dict):
        krum_param_m = 10

        def _compute_krum_score(vec_grad_list, byzantine_client_num):
            krum_scores = []
            num_client = len(vec_grad_list)
            for i in range(0, num_client):
                dists = []
                for j in range(0, num_client):
                    if i != j:
                        dists.append(
                            torch.norm(vec_grad_list[i] - vec_grad_list[j])
                            .item() ** 2
                        )
                dists.sort()  # ascending
                score = dists[0: num_client - byzantine_client_num - 2]
                krum_scores.append(sum(score))
            return krum_scores

        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        agent_updates_list = list(agent_updates_dict.values())
        # Compute list of scores
        krum_scores = _compute_krum_score(agent_updates_list, self.args.num_corrupt)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: krum_param_m]

        print('%d clients are selected' % len(score_index))
        return_updates = [agent_updates_list[i] for i in score_index]

        return sum(return_updates) / len(return_updates)

    def compute_robustLR(self, agent_updates_dict):

        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask = torch.zeros_like(sm_of_signs)
        mask[sm_of_signs < self.args.theta] = 0
        mask[sm_of_signs >= self.args.theta] = 1
        sm_of_signs[sm_of_signs < self.args.theta] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
        return sm_of_signs.to(self.args.device), mask

    def agg_mul_metric(self, agent_updates_dict, global_model, flat_global_model):
        # This is a simplified version and may need further refinement based on the original paper.
        vectorize_nets = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]

        cos_sim = smp.cosine_similarity(vectorize_nets)
        manhattan_dist = smp.manhattan_distances(vectorize_nets)
        euclidean_dist = smp.euclidean_distances(vectorize_nets)

        # Simple scoring based on sum of distances/similarities
        cos_scores = np.sum(cos_sim, axis=1)
        manhattan_scores = np.sum(manhattan_dist, axis=1)
        euclidean_scores = np.sum(euclidean_dist, axis=1)

        # Combine scores (simple average)
        scores = (cos_scores + manhattan_scores + euclidean_scores) / 3.0

        # Filter based on scores
        num_to_keep = len(vectorize_nets) - self.args.num_corrupt
        topk_ind = np.argsort(scores)[:num_to_keep]

        current_dict = {}
        client_keys = list(agent_updates_dict.keys())
        for idx in topk_ind:
            current_dict[client_keys[idx]] = agent_updates_dict[client_keys[idx]]

        update = self.agg_avg(current_dict)
        return update

    def agg_foolsgold(self, agent_updates_dict):
        # Simplified implementation of FoolsGold
        client_updates_list = [update.cpu().numpy() for update in agent_updates_dict.values()]
        n_clients = len(client_updates_list)

        cs = smp.cosine_similarity(client_updates_list) - np.eye(n_clients)
        max_cs = np.max(cs, axis=1)

        # Pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if max_cs[i] < max_cs[j]:
                    cs[i, j] = cs[i, j] * max_cs[i] / max_cs[j]

        wv = 1 - np.max(cs, axis=1)
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        # Rescale
        if np.max(wv) > 0:
            wv = wv / np.max(wv)

        wv[(wv == 1)] = 0.99
        wv = np.log(wv / (1 - wv)) + 0.5
        wv[np.isinf(wv)] = 1
        wv[wv < 0] = 0

        # Weighted aggregation
        weighted_updates = [update * w for update, w in zip(agent_updates_dict.values(), wv)]
        aggregated_update = torch.mean(torch.stack(weighted_updates, dim=0), dim=0)

        return aggregated_update