"""
Evaluation metrics and analysis tools.
"""

import numpy as np
from scipy.stats import t, ttest_ind

def mean_confidence_interval(data, confidence=0.95):
    arr = np.array(data)
    n = len(arr)
    mean_val = np.mean(arr)
    se = np.std(arr, ddof=1)/np.sqrt(n) if n>1 else 0
    h = se*t.ppf((1+confidence)/2., n-1) if (n>1) else 0
    return mean_val, mean_val-h, mean_val+h

def cohens_d(a, b):
    a = np.array(a)
    b = np.array(b)
    nx, ny = len(a), len(b)
    dof = nx+ny-2
    mean_x, mean_y = np.mean(a), np.mean(b)
    var_x, var_y = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((nx-1)*var_x+(ny-1)*var_y)/dof) if dof>0 else 1
    return (mean_x-mean_y)/pooled_std

class ResultsAnalyzer:
    def __init__(self, sota_acc=0.90):
        self.sota_acc = sota_acc

    def analyze_runs(self, accuracies):
        arr = np.array(accuracies)
        n = len(arr)
        mean_val = np.mean(arr)
        se = np.std(arr, ddof=1)/np.sqrt(n) if n>1 else 0
        h = se*t.ppf(0.975, n-1) if (n>1) else 0
        return {
            "mean": mean_val,
            "ci_low": mean_val - h,
            "ci_high": mean_val + h,
            "std": np.std(arr, ddof=1) if n>1 else 0
        }

    def significance_vs_sota(self, accuracies):
        arr = np.array(accuracies)
        n = len(arr)
        if n<2:
            return 0, 1.0
        sample_mean = np.mean(arr)
        sample_std = np.std(arr, ddof=1)
        se = sample_std/np.sqrt(n)
        tstat = (sample_mean-self.sota_acc)/se
        df = n-1
        pval = (1.-t.cdf(abs(tstat), df))*2
        return tstat, pval

    def compare_two_methods(self, a_list, b_list):
        tstat, pval = ttest_ind(a_list, b_list, equal_var=False)
        dval = cohens_d(a_list, b_list)
        return tstat, pval, dval

class GeometricAnalyzer:
    def analyze_dataset(self, dataset, out_csv, device="cpu", n=4):
        results = [["sample_idx","hausdorff_mean","hausdorff_ci_low","hausdorff_ci_high",
                   "chamfer_mean","chamfer_ci_low","chamfer_ci_high",
                   "norm_mean","norm_ci_low","norm_ci_high"]]
        hd_list = []
        cd_list = []
        nc_list = []
        n = min(n, len(dataset))
        for i in range(n):
            vol, _ = dataset[i]
            vol = vol.to(device, dtype=torch.float)
            shift = torch.roll(vol, shifts=1, dims=-1)
            hd = compute_hausdorff(vol[0], shift[0])
            cd = compute_chamfer(vol[0], shift[0])
            nc = compute_normal_consistency(vol[0], shift[0])
            hd_list.append(hd)
            cd_list.append(cd)
            nc_list.append(nc)
        hd_m, hd_l, hd_h = mean_confidence_interval(hd_list)
        cd_m, cd_l, cd_h = mean_confidence_interval(cd_list)
        nc_m, nc_l, nc_h = mean_confidence_interval(nc_list)
        results.append([0, hd_m, hd_l, hd_h, cd_m, cd_l, cd_h, nc_m, nc_l, nc_h])
        with open(out_csv, 'w') as f:
            for row in results:
                f.write(",".join(map(str,row))+"\n")
