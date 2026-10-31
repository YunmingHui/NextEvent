import argparse
from pathlib import Path
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser('preprocess the SNAP Bitcoin trust networks')
parser.add_argument('--dataset_name', type=str, required=True, choices=['bitcoinalpha', 'bitcoinotc'])
parser.add_argument('--node_feat_dim', type=int, default=172)
args = parser.parse_args()
name = args.dataset_name

df = pd.read_csv(f'../DG_data/{name}/{name}.csv', header=None, names=['src', 'dst', 'rating', 'ts'])
assert df.notna().all().all() and len(df.columns) == 4
num_raw_edges = len(df)

if name == 'bitcoinalpha':
    # US Eastern midnight -> start of that UTC day (05:00 / 04:00 UTC are both mapped to 00:00 UTC of the same date)
    assert np.all(df.ts % 3600 == 0) and set(((df.ts % 86400) // 3600).unique()) <= {4, 5}
    df['ts'] = (df.ts // 86400) * 86400

df = df.sort_values('ts', kind='mergesort').reset_index(drop=True)  # stable: keeps the raw order within a timestamp
df['ts'] = (df.ts - df.ts.min()).astype(np.float64)

raw_ids = np.union1d(df.src.values, df.dst.values)
id_map = {int(r): k + 1 for k, r in enumerate(raw_ids)}  # 0 stays the padding node
out = pd.DataFrame({'u': df.src.map(id_map).values, 'i': df.dst.map(id_map).values, 'ts': df.ts.values,
                    'label': np.zeros(len(df)), 'idx': np.arange(1, len(df) + 1)})
edge_feats = np.vstack([np.zeros((1, 1)), (df.rating.values / 10.0).reshape(-1, 1)])  # row 0: unused edge id 0
node_feats = np.zeros((len(raw_ids) + 1, args.node_feat_dim))

assert len(out) == num_raw_edges and np.all(np.diff(out.ts.values) >= 0) and out[['u', 'i']].values.min() == 1
assert np.array_equal(np.union1d(out.u, out.i), np.arange(1, len(raw_ids) + 1))
out_dir = Path(f'../processed_data/{name}')
out_dir.mkdir(parents=True, exist_ok=True)
out.to_csv(out_dir / f'ml_{name}.csv')
np.save(out_dir / f'ml_{name}.npy', edge_feats)
np.save(out_dir / f'ml_{name}_node.npy', node_feats)
print(f'{name}: {len(out)} edges, {len(raw_ids)} nodes (ids remapped to 1..{len(raw_ids)}), '
      f'{out.ts.nunique()} distinct timestamps, span {out.ts.max() / 86400:.1f} days, edge feature dim 1 (rating / 10)')
