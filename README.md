# SVV

> Recommender systems often suffer from noisy interactions like accidental clicks or popularity bias. Existing denoising methods typically identify users' intent in their interactions, and filter out noisy interactions that deviate from the assumed intent. However, they ignore that interactions deemed noisy could still aid model training, while some ``clean'' interactions offer little learning value. To bridge this gap, we propose Shapley Value-driven Valuation (SVV), a framework that evaluates interactions based on their objective impact on model training rather than subjective intent assumptions. In SVV, a real-time Shapley value estimation method is devised to quantify each interaction's value based on its contribution to reducing training loss. Afterward, SVV highlights the interactions with high values while downplaying low ones to achieve effective data pruning for recommender systems. In addition, we develop a simulated noise protocol to examine the performance of various denoising approaches systematically. Experiments on four real-world datasets show that SVV outperforms existing denoising methods in both accuracy and robustness. Further analysis also demonstrates that our SVV can preserve training-critical interactions and offer interpretable noise assessment. This work shifts denoising from heuristic filtering to principled, model-driven interaction valuation.


## Overall
Pytorch implementation for paper "Shapley Value-driven Data Pruning for Recommender Systems" published on KDD 2025.

## Requirements
- Python 3.9
- pytorch 1.13.0
- cuda 11

## Instruction
1. You may download Amazon Review dataset from https://nijianmo.github.io/amazon/index.html.

2. We provide an example on "CDs and Vinyl" datasets. The pre-processing data code `data_pre.py` can be founded in `data_pre` folder. And the example processed dataset for CDs in the `datasets` folder.

4. To set the python path, under the project root folder, run:
    ```
    source setup.sh
    ```
5. To train the base recommender: run:
    ```
    python scripts/train_base_amazon.py
    ```
6. To run fastshap method, run:
    ```
    python scripts/svv_fastshap.py
    ```

7. To run svv method, run:
   ```
   python scripts/svv.py
   ```


## Citation
Please cite our paper if you use this code in your work:
```latex
@article{zhang2025kdd,
  title={Shapley Value-driven Data Pruning for Recommender Systems},
  author={Zhang, Yansen and Zhang, Xiaokun, and Cui, Ziqiang and Ma, Chen},
  journal={KDD},
  year={2025}
}
```

## Acknowledgements

Thanks for this repos when developing this one:

[fastshap](https://github.com/iancovert/fastshap)
