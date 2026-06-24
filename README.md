# [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112)
### Python source code

We recommend reading [our blog](http://www.hyperbolicdeeplearning.com/) for an introduction to hyperbolic neural networks. Other related material can be accessed [here](https://people.csail.mit.edu/oct).

Source lives in `src/services/hyperbolic-nn/`.

## Docs (HTML)

Serve `./docs/index.html` locally (no image build; nginx pulls on first run):

```bash
cp .env.example .env
docker compose up docs-server
```

Open [http://localhost:8080](http://localhost:8080) (override with `DOCS_PORT` in `.env`).

## Docker (recommended)

```bash
cp .env.example .env
docker compose build hyperbolic-nn
docker compose run --rm hyperbolic-nn python -m pytest -p no:cacheprovider /test -q
```

Smoke training example (outputs go to `./output/` via `OUTPUT_DIR`):

```bash
docker compose run --rm hyperbolic-nn python hyp_rnn.py --config=configs/smoke.yaml
```

Override any YAML value from the CLI:

```bash
docker compose run --rm hyperbolic-nn python hyp_rnn.py \
  --config=configs/smoke.yaml \
  --device=cpu
```

MNIST sanity check (full train, expect ~0.92 test accuracy):

```bash
docker compose run --rm hyperbolic-nn python mnist_sanity.py \
  --data_dir=/jobs/datasets \
  --output_dir=/jobs \
  --device=cuda
```

Fast MNIST smoke:

```bash
docker compose run --rm hyperbolic-nn python mnist_sanity.py \
  --data_dir=/jobs/datasets \
  --output_dir=/jobs \
  --device=cuda \
  --epochs=1 \
  --train_samples=256 \
  --test_samples=128 \
  --max_train_batches=2 \
  --max_eval_batches=1 \
  --hidden_dim=16 \
  --batch_size=64
```

## Bare metal

1. Prerequisites:
```
Python 3.11+, PyTorch 2.12.1, TorchVision, numpy, matplotlib, tensorboard
```

2. Generate the 3d MLR figure from our paper.
```
python src/services/hyperbolic-nn/viz_mlr.py
```

3. Run the code to reproduce results from Table 1. Example using the PRFX10 hyperbolic GRU config:

```
python src/services/hyperbolic-nn/hyp_rnn.py --config=src/services/hyperbolic-nn/configs/prfx10_gru_hyp.yaml --root_path=./src/services/hyperbolic-nn/
```

The data needed in this code lives in the *_dataset folders and was generated as follows:

- SNLI data was put in a binary format using the file `binarize_snli_dataset.py` and the original [SNLI dataset](https://nlp.stanford.edu/projects/snli/)

- the PREFIX dataset was generated using the file `prefix_dataset.py`



## References
If you find this code useful for your research, please cite the following paper in your publication:
```
@inproceedings{ganea2018hyperbolic,
  title={Hyperbolic neural networks},
  author={Ganea, Octavian and B{\'e}cigneul, Gary and Hofmann, Thomas},
  booktitle={Advances in neural information processing systems},
  pages={5345--5355},
  year={2018}
}
```
