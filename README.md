# NextEvent: A Time-Driven Evaluation Framework for Temporal Graph Learning

The code is used for the paper "NextEvent: A Time-Driven Evaluation Framework for Temporal Graph Learning" submitted to the Web Conference 2027.

## Requirements

Python 3.11 and following packages

```bash
pip install torch numpy pandas scikit-learn tqdm torch_geometric tabulate
```

## Data

Download the raw data and put it in `DG_data/<dataset>/`:

- **Wikipedia, Reddit, MOOC, UCI**: from DyGLib's [Zenodo record](https://zenodo.org/records/7213796).
- **Bitcoin-OTC**: from SNAP [soc-sign-bitcoin-otc](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html).

Then preprocess:

```bash
cd preprocess_data
python preprocess_data.py --dataset_name wikipedia
python preprocess_bitcoin.py --dataset_name bitcoinotc
cd ..
```

## Run

Link prediction:

```bash
python train_link_prediction.py --dataset_name wikipedia --model_name TGN --seed 0 --load_best_configs --gpu 0
```

When prediction:

```bash
python train_node_activity_tpp.py --dataset_name wikipedia --model_name TGN --seed 0 --load_best_configs --gpu 0

# LNM-RNN baseline
python train_node_activity_tpp.py --dataset_name wikipedia --model_name LogNormMix --context_source rnn --seed 0 --gpu 0
```

`--model_name`: `TGN`, `CTAN`, `TGAT`, `GraphMixer`, `DyGFormer` (and `LogNormMix` for the when task).

Results are saved to `saved_results/`, checkpoints to `saved_models/`, logs to `logs/`.

* The code is built on [DyGLib](https://github.com/yule-BUAA/DyGLib). Two tasks: