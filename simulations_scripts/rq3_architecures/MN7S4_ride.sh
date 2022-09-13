 #!/bin/bash

nohup python3 -m scripts.train --model MN7S4_r_rapidarch_im005_ent00005_1 --seed 1 --save-interval 10 --frames 20000000  --env 'MiniGrid-MultiRoom-N7-S4-v0' --intrinsic-motivation 0.05 --im-type 'ride' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 1  --use-gpu 1  --gpu-id 1&
nohup python3 -m scripts.train --model MN7S4_r_rapidarch_im005_ent00005_2 --seed 2 --save-interval 10 --frames 20000000  --env 'MiniGrid-MultiRoom-N7-S4-v0' --intrinsic-motivation 0.05 --im-type 'ride' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 1  --use-gpu 1  --gpu-id 1&
nohup python3 -m scripts.train --model MN7S4_r_rapidarch_im005_ent00005_3 --seed 3 --save-interval 10 --frames 20000000  --env 'MiniGrid-MultiRoom-N7-S4-v0' --intrinsic-motivation 0.05 --im-type 'ride' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 1  --use-gpu 1  --gpu-id 1&
