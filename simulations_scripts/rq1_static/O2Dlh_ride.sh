 #!/bin/bash

nohup python3 -m scripts.train --model O2Dlh_r_im005_ent00005_1 --seed 1 --save-interval 10 --frames 50000000  --env 'MiniGrid-ObstructedMaze-2Dlh-v0' --intrinsic-motivation 0.05 --im-type 'ride' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 0  --use-gpu 1  --gpu-id 0&
nohup python3 -m scripts.train --model O2Dlh_r_im005_ent00005_2 --seed 2 --save-interval 10 --frames 50000000  --env 'MiniGrid-ObstructedMaze-2Dlh-v0' --intrinsic-motivation 0.05 --im-type 'ride' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 0  --use-gpu 1  --gpu-id 0&
nohup python3 -m scripts.train --model O2Dlh_r_im005_ent00005_3 --seed 3 --save-interval 10 --frames 50000000  --env 'MiniGrid-ObstructedMaze-2Dlh-v0' --intrinsic-motivation 0.05 --im-type 'ride' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 0  --use-gpu 1  --gpu-id 0&
