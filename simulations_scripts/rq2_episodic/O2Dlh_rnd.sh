 #!/bin/bash

nohup python3 -m scripts.train --model O2Dlh_rn_ep_im005_ent00005_1 --use-episodic-counts 1 --seed 1 --save-interval 10 --frames 50000000  --env 'MiniGrid-ObstructedMaze-2Dlh-v0' --intrinsic-motivation 0.05 --im-type 'rnd' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 0  --use-gpu 1  --gpu-id 0&
nohup python3 -m scripts.train --model O2Dlh_rn_ep_im005_ent00005_2 --use-episodic-counts 1 --seed 2 --save-interval 10 --frames 50000000  --env 'MiniGrid-ObstructedMaze-2Dlh-v0' --intrinsic-motivation 0.05 --im-type 'rnd' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 0  --use-gpu 1  --gpu-id 0&
nohup python3 -m scripts.train --model O2Dlh_rn_ep_im005_ent00005_3 --use-episodic-counts 1 --seed 3 --save-interval 10 --frames 50000000  --env 'MiniGrid-ObstructedMaze-2Dlh-v0' --intrinsic-motivation 0.05 --im-type 'rnd' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 0  --use-gpu 1  --gpu-id 0&
