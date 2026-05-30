import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f, norm


def fit_pca_spc(X_raw, k, alpha=0.95):
    n, p = X_raw.shape
    sc = StandardScaler()
    Xs = sc.fit_transform(X_raw)
    pca = PCA(n_components=k, svd_solver='full', random_state=42)
    T = pca.fit_transform(Xs)
    P = pca.components_.T
    lam = pca.explained_variance_
    T2 = np.sum((T**2) / lam, axis=1)
    T2_UCL = (k * (n - 1) / (n - k)) * f.ppf(alpha, k, n - k)
    E = Xs - pca.inverse_transform(T)
    Q = np.sum(E**2, axis=1)
    pf = PCA(n_components=min(p, n - 1), svd_solver='full', random_state=42)
    pf.fit(Xs)
    re = pf.explained_variance_[k:]
    t1, t2, t3 = re.sum(), (re**2).sum(), (re**3).sum()
    h0 = 1 - (2 * t1 * t3) / (3 * t2**2)
    z = norm.ppf(alpha)
    Q_UCL = t1 * ((z * np.sqrt(2 * t2 * h0**2) / t1) + 1 + (t2 * h0 * (h0 - 1) / t1**2)) ** (1 / h0)
    zc = norm.ppf(0.975)
    mu_e = E.mean(0)
    sd_e = E.std(0, ddof=1)
    W = T / lam
    cT2 = Xs * (W @ P.T)
    mu_c = cT2.mean(0)
    sd_c = cT2.std(0, ddof=1)
    return dict(
        scaler=sc, pca=pca, k=k, scores=T, loadings=P, eigenvalues=lam,
        T2=T2, Q=Q, T2_UCL=T2_UCL, Q_UCL=Q_UCL, X_scaled=Xs, E=E,
        evr=pca.explained_variance_ratio_ * 100, feature_names=[],
        n_train=n,
        Qcontrib_LCL=mu_e - zc * sd_e, Qcontrib_UCL=mu_e + zc * sd_e,
        T2contrib_LCL=mu_c - zc * sd_c, T2contrib_UCL=mu_c + zc * sd_c,
    )


def monitor_new(model, X_new):
    sc = model['scaler']
    pca = model['pca']
    lam = model['eigenvalues']
    Xns = sc.transform(X_new)
    Tn = pca.transform(Xns)
    T2n = np.sum((Tn**2) / lam, axis=1)
    En = Xns - pca.inverse_transform(Tn)
    Qn = np.sum(En**2, axis=1)
    return dict(
        T2=T2n, Q=Qn,
        T2_flag=T2n > model['T2_UCL'],
        Q_flag=Qn > model['Q_UCL'],
        Xn_s=Xns, Tn=Tn, En=En,
    )


def compute_rmsecv(X_s, max_k, G=10):
    n, p = X_s.shape
    max_k = min(max_k, p, n - 1)
    idx = np.arange(n)
    sg = [idx[g::G] for g in range(G)]
    vg = [np.array([j]) for j in range(p)]
    press = np.zeros(max_k)
    rmsecv = np.zeros(max_k)
    for k in range(1, max_k + 1):
        PRESS = 0.0
        COUNT = 0
        for ti in sg:
            tri = np.setdiff1d(idx, ti)
            Xtr = X_s[tri]
            Xte = X_s[ti]
            pc = PCA(n_components=min(k, p), svd_solver='full', random_state=42)
            pc.fit(Xtr)
            P = pc.components_.T
            for mc in vg:
                oc = np.setdiff1d(np.arange(p), mc)
                kk = min(k, P[oc, :].shape[0] - 1)
                if kk < 1:
                    continue
                Th, *_ = np.linalg.lstsq(P[oc, :kk], Xte[:, oc].T, rcond=None)
                Xh = (Th.T) @ P[:, :kk].T
                r = Xte[:, mc] - Xh[:, mc]
                PRESS += np.sum(r**2)
                COUNT += r.size
        press[k - 1] = PRESS
        rmsecv[k - 1] = np.sqrt(PRESS / COUNT) if COUNT > 0 else np.inf
    return int(np.argmin(rmsecv)) + 1, press, rmsecv


def iterative_cleaning(X_raw, k_clean, alpha_clean, max_iter=10):
    mask = np.ones(len(X_raw), dtype=bool)
    log = []
    for it in range(1, max_iter + 1):
        X_it = X_raw[mask]
        n_it = len(X_it)
        sc_it = StandardScaler()
        Xs_it = sc_it.fit_transform(X_it)
        k_it = min(k_clean, Xs_it.shape[1] - 1, n_it - 2)
        res = fit_pca_spc(X_it, k_it, alpha_clean)
        flag = (res['T2'] > res['T2_UCL']) | (res['Q'] > res['Q_UCL'])
        n_rem = int(flag.sum())
        log.append(dict(Iter=it, Before=n_it, Removed=n_rem, After=n_it - n_rem))
        if n_rem == 0:
            break
        idx_c = np.where(mask)[0]
        mask[idx_c[flag]] = False
    return mask, pd.DataFrame(log)
