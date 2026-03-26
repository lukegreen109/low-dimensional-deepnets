import torch as th
import h5py
import numpy as np
import pandas as pd

from utils import *
from reparameterization import *
from utils.embed import lazy_embed


# ...existing code...
import os, json
# ...existing code...

def _mk_qp_from_labels(labels_override, nclasses=None, endpoint_labels_override=None):
    """Build qs (sqrt one-hot labels) and ps (sqrt endpoint) from provided labels."""
    qs, ps, labels = {}, {}, {}
    def onehot(y, K=None):
        y = np.asarray(y, dtype=np.int32)
        K = int(K) if K is not None else int(y.max() + 1)
        oh = np.zeros((y.size, K), dtype=np.float32)
        oh[np.arange(y.size), y] = 1.0
        return oh, K

    for split in ["train", "val"]:
        key = "yh" if split == "train" else "yvh"
        y = labels_override[split]
        oh, K = onehot(y, K=nclasses)
        q = np.sqrt(np.expand_dims(oh, axis=0))

        # Endpoint: either provided labels or uniform
        if endpoint_labels_override is not None:
            y_p = endpoint_labels_override[split]
            oh_p, Kp = onehot(y_p, K=nclasses or K)
            p = np.sqrt(np.expand_dims(oh_p, axis=0))
        else:
            p = np.sqrt(np.ones_like(q) / (nclasses or K))

        qs[key] = q
        ps[key] = p
        labels[key] = np.asarray(y, dtype=np.int32)
    return qs, ps, labels


def main(loc="results/models/loaded", name='', n=100, ts=None, loaded=False, log=False,
         data_args={'data': 'CIFAR10', 'aug': 'none', 'sub_sample': 0}, pdata=None,
         labels_override=None, endpoint_labels_override=None, nclasses=None):
    # If labels are provided, bypass dataset loading
    if labels_override is not None:
        qs, ps, labels = _mk_qp_from_labels(labels_override, nclasses, endpoint_labels_override)
    else:
        data = get_data(data_args)
        if pdata is not None:
            pdata = get_data(pdata)
        labels, qs, ps = {}, {}, {}
        for key in ["train", "val"]:
            k = "yh" if key == "train" else "yvh"
            try:
                y_ = np.array(data[key].targets, dtype=np.int32)
            except AttributeError:
                y_ = np.array(data[key].y, dtype=np.int32)
            y = np.zeros((y_.size, y_.max() + 1), dtype=np.float32)
            y[np.arange(y_.size), y_] = 1.0
            qs[k] = np.sqrt(np.expand_dims(y, axis=0))
            if pdata is None:
                K = int(nclasses) if nclasses is not None else (y_.max() + 1 if y_.size else 10)
                ps[k] = np.sqrt(np.ones_like(qs[k]) / K)
            else:
                y_ = np.array(pdata[key].targets, dtype=np.int32)
                y = np.zeros((y_.size, y_.max() + 1), dtype=np.float32)
                y[np.arange(y_.size), y_] = 1.0
                ps[k] = np.sqrt(np.expand_dims(y, axis=0))
            labels[k] = y_

    if ts is None:
        ts = np.linspace(0, 1, n + 1)[1:]
    else:
        n = len(ts)

    geodesic = []
    eps = 1e-12

    for i in range(len(ts)):
        r = dict(seed=0, bseed=-1, m=f"{name}geodesic", opt=f"{name}geodesic")
        r['t'] = ts[i]
        for key in ["yh", "yvh"]:
            r[key] = gamma(ts[i], ps[key], qs[key]) ** 2
            if log:
                r[key] = np.log(np.clip(r[key], eps, None))
            ekey = "e" if key == "yh" else "ev"
            fkey = "f" if key == "yh" else "fv"
            e = (np.argmax(r[key], -1) == labels[key]).astype(np.float32)
            r[ekey] = e.squeeze()
            errkey = "err" if key == "yh" else "verr"
            r[errkey] = 1.0 - e.mean()
            # use epsilon to avoid log(0)
            f = - (np.log(np.clip(r[key], eps, None)) * qs[key]).sum(-1)
            r[fkey] = f.squeeze()
            r[f'{fkey}avg'] = f.mean()
        geodesic.append(r)

    # Filename config dict (match original behavior)
    if loaded:
        geodesic_df = pd.DataFrame(geodesic).reindex(
            columns=[
                "seed","bseed","m","opt","t","e","ev","err","verr",
                "f","fv","favg","fvavg","bs","drop","aug","bn","lr","wd","yh","yvh"
            ],
            fill_value="na",
        )
        for key in ["yh", "yvh"]:
            geodesic_df[key] = geodesic_df.apply(lambda rr: rr[key].squeeze(), axis=1)
        d = dict(geodesic_df[["seed","bseed","aug","m","bn","drop","opt","bs","lr","wd"]].iloc[0])
    else:
        d = {k: geodesic[0].get(k, "na") for k in ["seed","bseed","aug","m","bn","drop","opt","bs","lr","wd"]}

    d["seed"] = 0
    d["bseed"] = -1
    d["bsel"] = "geodesic"  # add batch selection tag for grouping by bsel

    fn = f"{json.dumps(d).replace(' ', '')}.p"
    os.makedirs(loc, exist_ok=True)
    out_path = os.path.join(loc, fn)
    print(out_path)
    if loaded:
        th.save(geodesic_df, out_path)
    else:
        th.save(geodesic, out_path)


def get_projection():
    key = "yh"
    f = h5py.File(f"/home/ubuntu/ext_vol/inpca/inpca_results_all/w_{key}_all_geod.h5", 'r')
    dp = f['w'][-100:, :-100]
    f.close()

    r = th.load(f"/home/ubuntu/ext_vol/inpca/inpca_results_all/r_{key}_all.p")
    d_mean = r["w_mean"]

    xp = lazy_embed(d_mean=d_mean, dp=dp, evals=r["e"], evecs=r["v"], ne=3)
    r["extra_points"] = xp
    th.save(r, f"/home/ubuntu/ext_vol/inpca/inpca_results_all/r_{key}_all.p")


if __name__ == "__main__":
    # synthetic data
    # root = '/home/ubuntu/ext_vol/data/'
    # config_fn = '/home/ubuntu/ext_vol/inpca/configs/data/synthetic-fc-50-0.5.yaml'
    # data_args = get_configs(config_fn)
    # ts = np.linspace(0.0, 1, 100)
    # main(loc='results/models/sloppy-50', ts=ts, loaded=True, log=False,
    #     data_args=data_args)
    for i in range(3):
        root = '/home/ubuntu/ext_vol/data/'
        config_fn = '/home/ubuntu/ext_vol/inpca/configs/data/uniform.yaml' 
        with open(config_fn) as f:
            data = yaml.safe_load(f)
        data_args = get_configs(config_fn)
        data_args['fn'] = os.path.join(root, f'CIFAR10_uniform_{i}.p')
        ts = np.linspace(0.0, 1, 100)
        main(loc='results/models/corners', name=f'tocorner{i}_', ts=ts, loaded=True, log=False,
            data_args=data_args)
    # get_projection()
