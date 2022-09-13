# An Evaluation Study of Intrinsic Motivation Techniques Applied to Reinforcement Learning over Hard Exploration Environments

This is the pytorch implementation of [CD-MAKE 2022](https://cd-make.net/) paper <br>*An Evaluation Study of Intrinsic Motivation Techniques Applied to Reinforcement Learning over Hard Exploration Environments*,<br> which can be found either in [Springer](https://link.springer.com/chapter/10.1007/978-3-031-14463-9_13) or [Arxiv](https://arxiv.org/abs/2205.11184).

The implementation is based on the code provided in [`rl-starter-files`](https://github.com/lcswillems/rl-starter-files) and is built to carry out experiments in [`MiniGrid`](https://github.com/Farama-Foundation/MiniGrid). This work was developed to analyze the impact of intrinsic motivation techniques in hard-exploration/sparse-rewards environments where base RL-algorithms are not sufficient to learn these kind of tasks.

<p align="center"><img src="README-rsrc/doorkey.png"></p>

## Installation

1. Clone this repository.

2. Install the dependencies with *pip*, so that `gym-minigrid`(environment), `pytorch` and other necessary packages/libraries are installed:

```
pip3 install -r requirements.txt
```
**Note:** This code was accordingly modified from: https://github.com/lcswillems/rl-starter-files. The `torch_ac` holds the same structure but does not work as the original implementation. Nevertheless, the example of usage is almost straightforward.  


## Example of use

In the `simulation_scripts` folder are provided the necessary scripts to reproduce the results with different seeds and algorithms. An example of a single simulation is as follows:

```
python3 -m scripts.train --model KS3R3_c_im0005_ent00005_1 --seed 1  --save-interval 10 --frames 30000000  --env 'MiniGrid-KeyCorridorS3R3-v0' --intrinsic-motivation 0.005 --im-type 'counts' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 0```
```

The hyperparameters and different criterias can me modified by reference them directly from the command line (or by modifying the default values directly in the `scripts/train.py`). Some of the most important hyperparameters are:
*   `--env`: environment to be used
*   `--frames`: the number of frames/timesteps to be run
*   `--seed`: the seed used to reproduce results
*   `--im_type`: specifies which kind of intrinsic motivation module is going to be used to compute the intrinsic reward
*   `--intrinsic_motivation`: the intrinsic coefficient value
*   `--separated-networks`: used to determine if the actor-critic agent will be trained with a single two-head CNN architecture or with two independent networks
*   `--model`: the directory where the logs and the models are saved

For example, setting the `--intrinsic_motivation 0` means that the agent will be trained without intrinsic rewards.


## Cite This Work

```
@InProceedings{10.1007/978-3-031-14463-9_13,
author="Andres, Alain
and Villar-Rodriguez, Esther
and Del Ser, Javier",
editor="Holzinger, Andreas
and Kieseberg, Peter
and Tjoa, A. Min
and Weippl, Edgar",
title="An Evaluation Study of Intrinsic Motivation Techniques Applied to Reinforcement Learning over Hard Exploration Environments",
booktitle="Machine Learning and Knowledge Extraction",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="201--220",
isbn="978-3-031-14463-9"
}
```

or:

```
@misc{https://doi.org/10.48550/arxiv.2205.11184,
  doi = {10.48550/ARXIV.2205.11184},
  url = {https://arxiv.org/abs/2205.11184},
  author = {Andres, Alain and Villar-Rodriguez, Esther and Del Ser, Javier},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {An Evaluation Study of Intrinsic Motivation Techniques applied to Reinforcement Learning over Hard Exploration Environments},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

