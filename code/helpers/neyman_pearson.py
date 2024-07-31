import torch
import numpy as np
import tqdm
from einops import rearrange
from types import SimpleNamespace

from scipy.stats import multivariate_normal, norm
from scipy.interpolate import interp1d


def dist2sim_(dists, slope=0.5):
    return 1.0 / (1.0 + dists * slope)

def estimate_neyman_pearson_task(labels_torch, logits_torch, nm_dists_torch, NUM_CLASSES, grid_num_samples=2000, dist2sim=dist2sim_):
    logits = logits_torch.cpu().numpy()
    nm_sim = dist2sim(nm_dists_torch.cpu().numpy())
    labels = labels_torch.cpu().numpy()

    # sample point on grid in the logits X dist space
    # assuming that dist is normalized to (0,1), e.g. using the dist2sim fnc
    grid_data = np.meshgrid(np.linspace(-np.max(logits) - 5, np.max(logits) + 5, num=grid_num_samples), np.linspace(0.0, 1, num=grid_num_samples))
    grid_data_rearrange = rearrange(np.stack(grid_data, axis=-1), "r c d -> (r c) d")

    logits_scales = []
    nm_sim_scales = []
    for c in range(0, NUM_CLASSES):
        mask = labels == c
        logits_scales.append(np.quantile(logits[mask, c], 0.9))
        nm_sim_scales.append(np.quantile(nm_sim[mask, c], 0.9))
    simulated_data = np.stack([norm.rvs(loc=0, scale=np.mean(logits_scales)/4.0, size=10000), 
                               norm.rvs(loc=0, scale=np.mean(nm_sim_scales)/8.0, size=10000)], axis=1)
    # NOTE: quantile of logits turns to be negative, because of the imbalance of class samples
    # simulated_data = np.stack([norm.rvs(loc=0, scale=np.quantile(logits, 0.9)/4.0, size=10000), 
    #                            norm.rvs(loc=0, scale=np.quantile(nm_sim, 0.9)/8.0, size=10000)], axis=1)
    # simulated_data = np.stack([norm.rvs(loc=0, scale=(np.quantile(logits, 0.9)-np.quantile(logits, 0.1))/4.0, size=10000), 
    #                            norm.rvs(loc=0, scale=(np.quantile(nm_sim, 0.9)-np.quantile(nm_sim, 0.1))/8.0, size=10000)], axis=1)
    print ("N-P log: logits min/max/q1/q9", np.min(logits), np.max(logits), np.quantile(logits, 0.1), np.quantile(logits, 0.9))
    print ("N-P log: nm_sim min/max/q1/q9", np.min(nm_sim), np.max(nm_sim), np.quantile(nm_sim, 0.1), np.quantile(nm_sim, 0.9))
    print ("N-P log: logits/nm_sim per-class scales", np.mean(logits_scales), np.mean(nm_sim_scales))
    s_cov = np.cov(simulated_data.T)

    class_means = []
    class_covs = []
    ood_estimate_means = []
    ood_estimate_covs = []
    ood_classes = []
    interp_fnc = []

    print("N-P Task: Estimating distributions parameters.")
    for c in tqdm.tqdm(range(0, NUM_CLASSES)):
        mask = labels == c
        if np.sum(mask) == 0:
            raise RuntimeError(f"ERROR(N-P Task): No datapoints for class {c}!")
            
        # [B, 2]
        id_space = np.stack([logits[mask, c], nm_sim[mask, c]], axis=1)
        # DEFAULT
        # [2]
        class_means.append(np.mean(id_space, axis=0))
        # [2, 2]
        class_covs.append(np.cov(id_space.T))

        # exp distribution
        # # [2]
        # class_means.append(np.array([np.max(id_space[:, 0]), np.min(id_space[:, 1])]))
        # # [2, 2]
        # class_covs.append(np.array(
        #     [
        #         (id_space.shape[0] - 2) / np.sum(class_means[-1][0] - id_space[:, 0]),
        #         (id_space.shape[0] - 2) / np.sum(id_space[:, 1] - class_means[-1][1]),
        #     ]))

        # guess
        ood_estimate_means.append(np.array([0, 0]))
        ood_estimate_covs.append(s_cov.copy())
        
        # other classes
        # ood_space = np.stack([logits[~mask, c], nm_sim[~mask, c]], axis=1)
        # gmm = GaussianMixture(n_components=NUM_CLASSES - 1, random_state=42).fit(ood_space)
        # ood_classes.append(gmm)

        # p_id_min = np.min(multivariate_normal.pdf(id_space, mean=class_means[-1], cov=class_covs[-1], allow_singular=True))
        # for cc in range(0, NUM_CLASSES):
        #     if cc != c:
        #         mask = labels == cc
        #         # [B, 2]
        #         ood_space = np.stack([logits[mask, c], nm_sim[mask, c]], axis=1)
        #         p_ood = multivariate_normal.pdf(ood_space, mean=class_means[-1], cov=class_covs[-1], allow_singular=True)
        #         mask = p_ood <= p_id_min
        #         if np.sum(mask) > ood_space.shape[1]:
        #             # [2]
        #             ood_classes.append((np.mean(ood_space[mask, :], axis=0), np.cov(ood_space[mask, :].T)))
        #         # ood_classes.append((np.mean(ood_space, axis=0), np.cov(ood_space.T)))

        p_c = multivariate_normal.pdf(grid_data_rearrange, mean=class_means[-1], cov=class_covs[-1], allow_singular=True)
        p_ood = multivariate_normal.pdf(grid_data_rearrange, mean=ood_estimate_means[-1], cov=ood_estimate_covs[-1], allow_singular=True)
        # p_c = class_covs[-1][0] * np.exp(-class_covs[-1][0]*np.clip(class_means[-1][0] - grid_data_rearrange[:,0], a_min=0, a_max=None)) *\
        #       class_covs[-1][1] * np.exp(-class_covs[-1][1]*np.clip(grid_data_rearrange[:, 1] - class_means[-1][1], a_min=0, a_max=None))  



        # p_ood_class = np.exp(gmm.score_samples(grid_data_rearrange))
        # p_ood = np.max(np.stack([p_ood, p_ood_class], axis=-1), axis=-1)
        # norm_pdf = 1.0 #/ (len(ood_classes) + 1)
        # p_ood = norm_pdf * np.max(np.stack([
        #     p_ood,
        #     *[multivariate_normal.pdf(grid_data_rearrange, mean=ood_classes[cc][0], cov=ood_classes[cc][1], allow_singular=True)
        #       for cc in range(0, len(ood_classes))]
        #     ], axis=-1), axis=-1)

        # p_c_log = multivariate_normal.logpdf(grid_data_rearrange, mean=class_means[-1], cov=class_covs[-1], allow_singular=True)
        # p_ood = multivariate_normal.logpdf(grid_data_rearrange, mean=ood_estimate_means[-1], cov=ood_estimate_covs[-1], allow_singular=True)

        # p_ood_class = gmm.score_samples(grid_data_rearrange)
        # p_ood = np.max(np.stack([p_ood, p_ood_class], axis=-1), axis=-1)

        r = p_c / p_ood
        # r = p_c_log - p_ood
        rs_id = np.argsort(r) 
        y_val = []
        x_val = []
        for i in range(1, rs_id.shape[0]):
            if r[rs_id[i]] == r[rs_id[i-1]]:
                continue
            thr = 0.5*(r[rs_id[i]] + r[rs_id[i-1]])
            x_val.append(thr)
            if len(y_val) > 0:
                y_val.append(y_val[-1] + p_c[rs_id[i]])
            else:
                y_val.append(p_c[rs_id[i]])
        y_val = np.array(y_val) / np.sum(p_c)  
        interp_fnc.append(interp1d([0.0] + x_val + [np.Inf], [0] + y_val.tolist() + [1.0], kind="linear"))
        # interp_fnc.append(interp1d([-np.Inf] + x_val + [0.0], [0] + y_val.tolist() + [1.0], kind="linear"))

    return SimpleNamespace(class_means=class_means, 
                           class_covs=class_covs,
                           ood_estimate_means=ood_estimate_means,  
                           ood_estimate_covs=ood_estimate_covs,
                           ood_classes=ood_classes,
                           interp_fnc=interp_fnc)

def eval_neyman_pearson_task(n_p_model, logits_torch, nm_dists_torch, NUM_CLASSES, dist2sim=dist2sim_):
    logits = logits_torch.cpu().numpy()
    nm_sim = dist2sim(nm_dists_torch.cpu().numpy())

    eval_pdf = np.zeros_like(nm_sim)
    for c in range(0, NUM_CLASSES):
        eval_space = np.stack([logits[:, c], nm_sim[:, c]], axis=1)
        p_x1 = multivariate_normal.pdf(eval_space, mean=n_p_model.class_means[c], cov=n_p_model.class_covs[c])
        p_x2 = multivariate_normal.pdf(eval_space, mean=n_p_model.ood_estimate_means[c], cov=n_p_model.ood_estimate_covs[c])

        # p_x1 = n_p_model.class_covs[c][0] * np.exp(-n_p_model.class_covs[c][0]*np.clip(n_p_model.class_means[c][0] - eval_space[:,0], a_min=0, a_max=None)) *\
        #       n_p_model.class_covs[c][1] * np.exp(-n_p_model.class_covs[c][1]*np.clip(eval_space[:, 1] - n_p_model.class_means[c][1], a_min=0, a_max=None))  


        # p_x2_class = np.exp(n_p_model.ood_classes[c].score_samples(eval_space))
        # p_x2 = np.max(np.stack([p_x2, p_x2_class], axis=-1), axis=-1)

        # norm_pdf = 1.0 #/ (len(n_p_model.ood_classes) + 1)
        # p_x2 = norm_pdf * np.max(np.stack([
        #     p_x2,
        #     *[multivariate_normal.pdf(eval_space, mean=n_p_model.ood_classes[cc][0], cov=n_p_model.ood_classes[cc][1], allow_singular=True)
        #       for cc in range(0, len(n_p_model.ood_classes))]
        #     ], axis=-1), axis=-1)

        # p_x1_log = multivariate_normal.logpdf(eval_space, mean=n_p_model.class_means[c], cov=n_p_model.class_covs[c])
        # p_x2 = multivariate_normal.logpdf(eval_space, mean=n_p_model.ood_estimate_means[c], cov=n_p_model.ood_estimate_covs[c])

        # p_x2_class = n_p_model.ood_classes[c].score_samples(eval_space)
        # p_x2 = np.max(np.stack([p_x2, p_x2_class], axis=-1), axis=-1)

        r = p_x1 / p_x2 
        # r = p_x1_log - p_x2 
        eval_pdf[:, c] = n_p_model.interp_fnc[c](r) 

    device = logits_torch.get_device()
    if device < 0:
        device = "cpu"
    return torch.from_numpy(eval_pdf).to(device)


