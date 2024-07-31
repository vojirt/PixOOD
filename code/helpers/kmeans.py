import torch
from einops import rearrange
import time
import numpy as np


class OnlineKMeans():
    def __init__(self, mahalanobis_dist=False, normalize_by_tau=False):
        self.MAX_NUM_CLUSTERS = 250
        self.normalize_by_tau = normalize_by_tau

        # [K, Dim]
        self.clusters = None
        # [K]
        self.cluster_var = None
        self.cluster_weight = None

        # [1]
        self.exp_decay = 0.01
        self.intra_dist_factor = 3.0
        self.max_var = 30**2
        self.min_var = 5**2

        # flags
        self.mahalanobis_dist = mahalanobis_dist

    def _dist_fnc(self, x, y, **kwargs):
        if self.mahalanobis_dist:
            # [D, D]
            cov = kwargs.get("cov").float()
            d = (x - y).float()

            # cov is not singular?
            if torch.linalg.det(cov) > 0.0:
                # [..., D]
                # [...]
                return ((d @ torch.linalg.inv(cov)) * d).sum(-1).sqrt()
            else:
                # what to do, just set all dists to "inf", this cluster will be removed in consolidation
                return 1e9 * torch.ones_like(d)[..., 0]
        else:
            max_data_size = (1000.0*1000) / y.shape[0]
            num_of_splits = np.ceil(x.shape[0]  / max_data_size).astype(int)
            if num_of_splits > 1: 
                x_chunks = torch.chunk(x, num_of_splits, dim=0)
                dist = []
                for xc in x_chunks:
                    # dist.append((xc - y).pow(2).sum(-1).sqrt())
                    dist.append(torch.cdist(xc[None, ...], y[None, ...], p=2)[0, ...])
                final_dist = torch.cat(dist, dim=0)
            else:
                # return (x - y).pow(2).sum(-1).sqrt()
                final_dist = torch.cdist(x[None, ...], y[None, ...], p=2)[0, ...]
            if hasattr(self, "normalize_by_tau") and self.normalize_by_tau:
                final_dist = final_dist / self.cluster_var[None, :]

            return final_dist

    def compute_dist(self, x):
        # x ... [N, Dim]
        # out ... [N, K]
        if self.mahalanobis_dist:
            d = []
            for i in range(0, self.clusters.shape[0]):
                d.append(self._dist_fnc(self.clusters[i:(i+1), :], x, cov=self.cluster_var[i, ...]))
            return torch.stack(d, dim=-1) 
        else:
            return self._dist_fnc(self.clusters[None, ...], x[:, None, :])

    def assignments(self, x):
        started_at = time.time()
        dists = self.compute_dist(x) 
        min_d, closest = torch.min(dists, dim=1)
        neg_mask_sigma = 3 

        cluster_assignmets = []
        if self.mahalanobis_dist:
            dist_mask = (min_d < self.intra_dist_factor)
            dist_mask_neg_all = (dists < neg_mask_sigma)       # within "3 sigma"
            cluster_assignmets = torch.nn.functional.one_hot(closest, num_classes=self.clusters.shape[0]).bool().logical_and(dist_mask[:, None])
            # unassigned points are those further than "x sigma" from clusters
            nonassigned = dist_mask_neg_all.sum(-1).bool().logical_not()

        else:
            cluster_assignmets_neg = []
            nonassigned = []
            for u in range(0, self.clusters.shape[0]):
                ids = (closest == u) & (min_d < self.intra_dist_factor*self.cluster_var[u].sqrt())
                ids_neg = (dists[:, u] < neg_mask_sigma*self.cluster_var[u].sqrt())

                cluster_assignmets.append(ids)
                cluster_assignmets_neg.append(ids_neg)

            # unassigned points are those further than "x sigma" from clusters
            nonassigned = torch.stack(cluster_assignmets_neg, dim=-1).sum(-1) == 0
            cluster_assignmets = torch.stack(cluster_assignmets, dim=-1)

        # print(f"assignments: {time.time() - started_at:0.3f} sec.; num_clusters: {self.clusters.shape[0]}; size of x: {x.shape[0]}")
        return cluster_assignmets.bool(), nonassigned

    def update_cluster(self, c_id, xc, w = 1.0):
        decay = self.exp_decay * w
        # exp. decay type of update of parameters
        self.clusters[c_id, :] = self.clusters[c_id, :] + decay * (xc - self.clusters[c_id:c_id+1, :]).mean(0)
        self.cluster_var[c_id] = (1.0 - decay) * self.cluster_var[c_id] + decay * self.var_estimation(xc, self.clusters[c_id:c_id+1, :], robust=False)
        self.cluster_weight[c_id] = (1.0 - decay) * self.cluster_weight[c_id] + decay * xc.shape[0]        

    def var_estimation(self, xc, mean, robust=False):
        device = "cpu" if xc.get_device() < 0 else "cuda"
        # [N, D]
        diff = (mean - xc)
        if robust:
            l2diff = diff.pow(2).sum(-1)
            tentative_std = [torch.quantile(l2diff, f/100.0).sqrt().item() for f in range(1, 90)]
            min_s = min(1.0, torch.quantile(l2diff, 0.1).sqrt().item())
            max_s = torch.quantile(l2diff, 0.7).sqrt().item()
            tentative_std = np.linspace(min_s, max_s, 1000)
            # print(f"Tentative std: {tentative_std}")
            tentative_likelihoods = []
            for std in tentative_std:
                l2diff_filtered = l2diff[(l2diff < (3*std)**2) & (l2diff > 0)]
                if l2diff_filtered.shape[0] > 1:
                    tentative_likelihoods.append((torch.exp(-0.5*l2diff_filtered / (std**2)) / (np.sqrt(2*torch.pi)*std)).sum().item() / l2diff_filtered.shape[0])
                else:
                    tentative_likelihoods.append(0.0)
            # print(f"Tentative likelihoods: {tentative_likelihoods}")
            idd = torch.argmax(torch.tensor(tentative_likelihoods))
            std = tentative_std[idd.item()]
            var = 3 * std ** 2
            # print(f"VarEstimation: selected idd {idd} with var {var:0.5f} (std {std:0.5f})")

            if self.mahalanobis_dist:
                # initialize as hyper-sphere, i.e. diagonal covariance with the same elements
                return torch.diag(var*torch.ones(xc.shape[-1], device=device))
                # return torch.diag(torch.quantile(diff.pow(2), 0.1)*torch.ones(xc.shape[-1], device=device))
            else:
                # return torch.quantile(diff.pow(2).sum(-1), 0.1)
                return torch.tensor(var, device=device)
        else:
            # sample variance (unbiased ... -1 in denominator)
            if self.mahalanobis_dist:
                # [N, D, 1] @ [N, 1, D] -> [N, D, D]
                # diff_cpu = diff.cpu()
                # prods = (diff_cpu[:, :, None] @ diff_cpu[:, None, :]).to(device)
                # return prods.sum(0) / (xc.shape[0] - 1.0)
                return torch.diag(diff.pow(2).sum(0) / (xc.shape[0] - 1.0)) 
            else:
                return diff.pow(2).sum(-1).sum() / (xc.shape[0] - 1.0)

    def add_cluster(self, xc, select_from=None):
        device = "cpu" if xc.get_device() < 0 else "cuda"
        if select_from is None:
            rndid = torch.randperm(xc.size(0))[0]
            new_c = xc[rndid:rndid+1, :]
        else:
            rndid = torch.randperm(select_from.size(0))[0]
            new_c = select_from[rndid:rndid+1, :]

        if self.clusters is None:
            # init
            self.clusters = new_c 
            if self.mahalanobis_dist:
                self.cluster_var = torch.empty((0, xc.shape[-1], xc.shape[-1]), device=device)
            else:
                self.cluster_var = torch.empty((0,), device=device)
            self.cluster_weight = torch.empty((0,), device=device)

            # robust initial quess of cluster size
            var = self.var_estimation(xc, self.clusters, robust=True)[None, ...]
            reestimate = False
        else:
            self.clusters = torch.cat([self.clusters, new_c], dim=0)
            # initial quess of cluster size as mean size of existing clusters
            if self.mahalanobis_dist:
                # var = torch.diag(torch.mean(self.cluster_var, dim=0).diag().mean() * torch.ones(self.cluster_var.shape[1], device=device))[None, ...]
                var = torch.mean(self.cluster_var, dim=0)[None, ...]
            else:
                var = torch.mean(self.cluster_var, dim=0, keepdim=True)
                # var = torch.cat([torch.mean(self.cluster_var, dim=0, keepdim=True), self.var_estimation(xc, self.clusters[-1:, :], robust=True)[None, ...]], dim=0).min(dim=0)[0][None, ...]
            # var = self.var_estimation(xc, self.clusters[-1:, :], robust=True)[None, ...]

            reestimate = False

        self.cluster_var = torch.cat([self.cluster_var, var], dim=0)

        # NOTE: here kind of assuming that the data have some structure (they are 
        #       coming from batch of images so it is fair to assume there will be similar pixels within images

        # re-estimate cluster size based on point assignments
        dists = self._dist_fnc(self.clusters[None, -1, ...], xc[:, None, :], cov=self.cluster_var[-1, ...])[:, 0]
        if self.mahalanobis_dist:
            mask = dists < self.intra_dist_factor 
        else:
            mask = dists < self.intra_dist_factor * self.cluster_var[-1].sqrt() 
        w = mask.sum()
        self.cluster_weight = torch.cat([self.cluster_weight, torch.tensor([w], device=device)], dim=0)

        if w > 1 and reestimate:
            self.cluster_var[-1] = self.var_estimation(xc[mask, :], self.clusters[-1, ...], robust=True)

    def add_batch(self, x, add_new=True):
        with torch.no_grad():
            # if init (clusters == None): add new cluster as mean value
            # else, find distances of mi to x
            device = "cpu" if x.get_device() < 0 else "cuda"
            if self.clusters is None:
                self.add_cluster(x)
            else:
                # [N, K]
                # replace closest mi by mi + a*( x - mi )
                cluster_assignments, nonassigned = self.assignments(x)
                for u in range(0, self.clusters.shape[0]):
                    if cluster_assignments[:, u].sum() > 1:
                        self.update_cluster(u, x[cluster_assignments[:, u], :])

                # any too far? --> add new mi
                # if len(nonassigned.shape) > 1 and nonassigned.shape[0] > 0:
                #     nonassigned = nonassigned[:, 0]
                if nonassigned.sum() > 1 and add_new and (self.clusters is not None and self.clusters.shape[0] < self.MAX_NUM_CLUSTERS):
                    self.add_cluster(x, select_from=x[nonassigned, :]) 
                    
            if self.cluster_var is not None:
                self.cluster_var = torch.clamp(self.cluster_var, self.min_var, self.max_var)

    def remove_clusters(self, to_remove):
        if len(to_remove) == self.clusters.shape[0]:
            self.clusters = None
            self.cluster_var = None
            self.cluster_weight = None
        else:
            self.clusters = self.clusters[[i for i in range(0, self.clusters.shape[0]) if i not in to_remove], :]
            self.cluster_var = self.cluster_var[[i for i in range(0, self.cluster_var.shape[0]) if i not in to_remove]]
            self.cluster_weight = self.cluster_weight[[i for i in range(0, self.cluster_weight.shape[0]) if i not in to_remove]]

    def consolidate(self):
        if self.clusters is None or self.clusters.shape[0] < 2:
            return

        to_remove = []
        total_weight = self.cluster_weight.sum()
        for i in range(0, self.clusters.shape[0]):
            # remove if cluster have less than 3 support point or less than 0.5% of total points
            if self.cluster_weight[i] < 3 or self.cluster_weight[i] < 0.005*total_weight:
                to_remove.append(i)
            # remove clusters with "singular" cov
            if self.mahalanobis_dist and torch.linalg.det(self.cluster_var[i, ...]) <= 0:
                to_remove.append(i)
                continue
            for j in range(0, self.clusters.shape[0]):
                if j != i and j not in to_remove:
                    if self.mahalanobis_dist:
                        # NOTE: only for diagonal covs
                        if torch.all(((self.clusters[i, :] - self.clusters[j, :]).pow(2) + self.cluster_var[i,...].diag() - self.cluster_var[j, ...].diag()) <= 0):
                            to_remove.append(i)
                    else:
                        if (self.clusters[i, :] - self.clusters[j, :]).pow(2).sum(-1) + self.cluster_var[i] <= self.cluster_var[j]:
                            to_remove.append(i)
        self.remove_clusters(to_remove)
        
    @property 
    def count(self):
        return 0 if self.clusters is None else self.cluster_var.shape[0]

    @property         
    def weight(self):
        return 0 if self.clusters is None else self.cluster_weight.sum()

    def nearest_intra_dist(self):
        dists = []
        for c in range(0, self.clusters.shape[0]):
            dists.append(self._dist_fnc(self.clusters[c:(c+1), ...], self.clusters, cov=self.cluster_var[c, ...]))
        dists = torch.stack(dists, dim=0)
        dists.fill_diagonal_(1e9)
        return dists.min(dim=-1)[0]

class DiffKMeansMultiClassv2(torch.nn.Module):
    def __init__(self, num_classes, K_per_class, emb_size, _data=None, _labels=None, init_tau=20.0, reset_assignment_thr=0.5, knn_type="condensation"):
        super(DiffKMeansMultiClassv2, self).__init__()
        self.num_classes = num_classes
        self.K_per_class = K_per_class
        self.emb_size = emb_size
        self.knn_type = knn_type

        # self.cluster_temp = torch.nn.Parameter(0.5*torch.ones((1, num_classes, 1)), requires_grad=False)
        self.cluster_temp = torch.nn.Parameter(0.5*torch.ones(1), requires_grad=False)

        # self.exp_distr_temp = torch.nn.Parameter((100/K_per_class)*torch.ones(num_classes), requires_grad=True)

        self.exp_distr_sigmoid_temp = 2
        self.exp_distr_sigmoid_max_value = 100
        # inverse sigmoid computation
        assert 0 < init_tau < self.exp_distr_sigmoid_max_value
        desired_tau = init_tau / self.exp_distr_sigmoid_max_value
        self.exp_distr_temp = torch.nn.Parameter(-self.exp_distr_sigmoid_temp
                                                 * torch.log(torch.tensor((1-desired_tau)/desired_tau))
                                                 * torch.ones(num_classes,
                                                              K_per_class),
                                                 requires_grad=True)

        self.reset_assignment_thr_ = reset_assignment_thr

        # whitening
        self.register_buffer("norm_data_med", torch.median(_data, dim=0)[0], persistent=True)
        self.register_buffer("norm_data_std", torch.std(_data, dim=0), persistent=True)

        self._mu = None 
        self.init_mu(self.norm_data(_data), _labels)

        # to check the cluster assignment counts
        self.decay_assign = 0.001
        self.class_t = torch.zeros(self.num_classes, dtype=torch.long)
        self.running_assignment = torch.nn.Parameter(torch.ones(num_classes, K_per_class), requires_grad=False)
        self.running_batchsize = torch.nn.Parameter(torch.ones(num_classes), requires_grad=False)
        self.hist_assignment = []
        self.training_loss = [[] for _ in range(0, self.num_classes)]

    def init_mu(self, data_mc, labels):
        device = "cuda" if data_mc.get_device() >= 0 else "cpu"
        if data_mc is None:
            self._mu = torch.nn.Parameter(torch.rand(self.num_classes, self.K_per_class, self.emb_size)-0.5, requires_grad=True)
        else:
            mu = torch.zeros(self.num_classes, self.K_per_class, self.emb_size).to(device)
            for c in range(0, self.num_classes):
                mask = (labels == c)
                if mask.sum() == 0: 
                    mu[c, ...] = torch.rand(self.K_per_class, self.emb_size).to(device) - 0.5
                else:
                    data = data_mc[mask, :]
                    if data.shape[0] > self.K_per_class: 
                        idx = np.random.choice(np.arange(data.shape[0]), size=self.K_per_class, replace=False)
                        noise = 0
                    else:
                        # if there is not enough data to reset, sample with replacement (i.e. multiple same indexes can be choosen)
                        idx = np.random.choice(np.arange(data.shape[0]), size=self.K_per_class)
                        noise = (0.1*(torch.rand(self.K_per_class, self.emb_size)-0.5)).to(device)
                    mu[c, ...] = data[idx, :] + noise
            self._mu = torch.nn.Parameter(mu, requires_grad=True)
        self._mu_init = self._mu.detach().cpu()

    def sim_fnc(self, data, tocpu=False):
        mu = rearrange(self._mu, "c k d -> (c k) d") 

        # estimated based on 12GB gpu and 1024 dim embeddings
        max_data_size = 2000 * 1000 / float(self.num_classes * self.K_per_class)
        num_of_splits = np.ceil(data.shape[0]  / max_data_size).astype(int)
        if num_of_splits > 1:
            data_chunks = torch.chunk(data, num_of_splits, dim=0)
            dist = []
            for dc in data_chunks:
                if tocpu:
                    dist.append((-torch.cdist(dc[None, ...], mu[None, ...], p=2)[0, ...]).cpu())
                else:
                    dist.append(-torch.cdist(dc[None, ...], mu[None, ...], p=2)[0, ...])
            dist = torch.cat(dist, dim=0)
        else:
            dist = -torch.cdist(data[None, ...], mu[None, ...], p=2)[0, ...]
        
        # because the data are normalize, scale the dist to some reasonable
        # range so that the softmax temp and exp tau works properly and are
        # initialized in "good enough" range
        dist *= 100.0/np.sqrt(data.shape[1])
        
        self.not_valid_mask = rearrange(torch.logical_not(self.mu_valid), "c k -> (c k)")
        dist[:, self.not_valid_mask] = -1e12
        
        return rearrange(dist, "b (c k) -> b c k", c=self.num_classes, k=self.K_per_class)
        
    def compute_assignment(self, sim):
        # soft assignment - cluster responsibilities via softmax
        # [N, C, K]
        r = torch.nn.functional.softmax(self.cluster_temp * sim, dim=2)
        return r

    def update_stats(self, r, labels):
        # [C, K]
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() == 0:   # no samples -> no update
                continue
            
            # hard assignment
            # assignment_count = torch.nn.functional.one_hot(torch.argmax(r[mask, c, :].detach(), dim=-1), num_classes=self._mu.shape[1]).float().sum(dim=0)
            # soft assignment
            assignment_count = r[mask, c, :].detach().sum(dim=0)

            # unbiased EWA decay estimate
            self.class_t[c] = self.class_t[c] + 1
            qt = self.decay_assign / (1 - torch.pow(1 - self.decay_assign, self.class_t[c]))
            
            # EWA
            self.running_assignment[c, :] = (1.0 - qt) * self.running_assignment[c, :] + qt * assignment_count
            # EWA
            self.running_batchsize[c] = (1.0 - qt) * self.running_batchsize[c] + qt * mask.sum().cpu().item()
        self.hist_assignment.append(self.running_assignment.detach().clone().cpu().numpy())

    def forward(self, data_dict):
        data = self.norm_data(data_dict.data)
        labels = data_dict.labels

        # [N, C, K]
        sim = self.sim_fnc(data)
        # [N, C, K]
        r = self.compute_assignment(sim)

        self.update_stats(r, labels)

        # pdf = self.eval_distribution(-sim) 
        logpdf = self.eval_log_distribution(-sim) 

        kmeans_loss = 0
        for c in range(0, self.num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue

            if self.knn_type == "condensation":
                # (PAPER) S-H Condensation 
                class_loss = -(logpdf[mask, c, :] * r[mask, c, :]).sum(dim=1).mean(dim=0)
            elif self.knn_type == "softkmeans":
                # soft k-means
                class_loss = ((-sim[mask, c, :]).pow(2) * r[mask, c, :]).sum(dim=1).mean(dim=0)
            elif self.knn_type == "softkmedians":
                # soft k-medians
                class_loss = ((-sim[mask, c, :]) * r[mask, c, :]).sum(dim=1).mean(dim=0)
            else:
                raise NotImplementedError

            # soft piece-wise ML condensation
            # class_loss = -((pdf[mask, c, :] * r[mask, c, :]).sum(dim=1) + 1e-12).log().mean(dim=0)

            # soft k-means scaled
            # class_loss = (((-sim[mask, c, :]).pow(2) / self.tau[c:c+1, :]) * r[mask, c, :]).sum(dim=1).mean(dim=0)

            self.training_loss[c].append(class_loss.detach().cpu().item())
            kmeans_loss += class_loss 

        return kmeans_loss
        
    def reset_clusters(self, data_dict):
        device = "cuda" if self._mu.get_device() >= 0 else "cpu"
        with torch.no_grad():
            data = self.norm_data(data_dict.data)
            labels = data_dict.labels

            for c in range(0, self.num_classes):
                to_reset_mask = self.running_assignment[c, :] < self.reset_assignment_thr[c]
                to_reset = torch.nonzero(to_reset_mask).flatten().cpu().numpy().tolist()
                valid_data = labels == c
                if len(to_reset) > 0 and valid_data.sum().item() > 0:
                    idx = np.random.choice(np.arange(valid_data.sum().item()), size=len(to_reset))
                    self._mu[c, to_reset, :] = data[valid_data, :][idx, :] + (1e-6/np.sqrt(self.emb_size))*torch.rand((len(to_reset), self.emb_size)).to(device)
                    self.running_assignment[c, to_reset] = 1.1*self.reset_assignment_thr[c]

    def assignment(self, data):
        # [N, C, K]
        sim = self.sim_fnc(data).detach()
        # [N, C, K]
        r = self.compute_assignment(sim)
        return sim, r
    
    @property
    def tau(self):
        # tau = torch.nn.functional.relu(self.exp_distr_temp[c:c+1, :]) + 0.01 
        # tau = torch.nn.functional.relu(self.exp_distr_temp) + 0.01 
        tau = torch.nn.functional.sigmoid(self.exp_distr_temp / self.exp_distr_sigmoid_temp) * self.exp_distr_sigmoid_max_value + 1./self.exp_distr_sigmoid_max_value 
        # tau = self.exp_distr_temp
        return tau

    def eval_distribution_sum_large(self, dist, qnorm=False):
        device = "cuda" if self._mu.get_device() >= 0 else "cpu"
        # estimated based on 12GB gpu and 1024 dim embeddings
        tau_inv = 1 / self.tau[None, ...]
        if qnorm:
            q = self.assignment_q[None, :] * tau_inv
        else:
            q = 1.0
        max_data_size = 8000 * 1000 / float(self.num_classes * self.K_per_class)
        num_of_splits = np.ceil(dist.shape[0]  / max_data_size).astype(int)
        if num_of_splits > 1:
            data_chunks = torch.chunk(dist, num_of_splits, dim=0)
            pdf = []
            for dc in data_chunks:
                pdf.append((q * torch.exp(-dc.to(device) * tau_inv)).sum(dim=-1).cpu())
            return torch.cat(pdf, dim=0) 
        else:
            return (torch.exp(-dist * tau_inv) * tau_inv).sum(dim=-1).cpu()

    def eval_distribution(self, dist):
        tau = self.tau
        # return torch.exp(-dist / tau[None, :, None]) / tau[None, :, None]
        return torch.exp(-dist / tau[None, ...]) / tau[None, ...]
        # return torch.exp(-dist.pow(2) / (2*tau[None, :, None].pow(2))) / (tau[None, :, None]*np.sqrt(2*np.pi))

    def eval_log_distribution(self, dist):
        tau = self.tau
        # return (-dist / tau[None, :, None]) - tau[None, :, None].log()
        return (-dist / tau[None, ...]) - tau[None, ...].log()
        # return (-dist.pow(2) / (2*tau[None, :, None].pow(2))) - (tau[None, :, None]*np.sqrt(2*np.pi)).log()

    @property
    def reset_assignment_thr(self):
        return (self.running_batchsize / self._mu.shape[1])*self.reset_assignment_thr_

    @property
    def assignment_q(self):
        return self.running_assignment / self.running_assignment.sum(dim=-1, keepdim=True)

    @property
    def mu_valid(self):
        return (self.running_assignment > self.reset_assignment_thr[:, None].tile(1, self.running_assignment.shape[1]))

    @property
    def mu(self):
        return (self._mu * self.norm_data_std[None, :]) + self.norm_data_med[None, :]

    def norm_data(self, data):
        return (data - self.norm_data_med[None, :]) / self.norm_data_std[None, :]


# ============== Helper functions ============== 
def majority_vote_patch_label_selection(target, patch_size):
    unique_labels = torch.unique(target)
    # [B, L, Hf, Wf]
    counts_per_label = torch.nn.functional.interpolate(
                            torch.nn.AvgPool2d((patch_size, patch_size))
                                (
                                   torch.stack([(target == ul).float() for ul in unique_labels], dim=1)
                                ), 
                            size=[target.shape[-2]//patch_size, target.shape[-1]//patch_size], mode="bilinear", align_corners=True) 

    # [B * Hf * Wf]
    weights, argmax_id = torch.max(counts_per_label, dim=1)
    labels = unique_labels[rearrange(argmax_id, "b h w -> (b h w)")]
    weights = rearrange(weights / torch.sum(counts_per_label, dim=1), "b h w -> (b h w)") 
    return labels, weights
