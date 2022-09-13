 #!/bin/bash

nohup python3 -m scripts.train --model MN7S8_c_ngu_im0005_ent00005_1  --seed 1 --save-interval 10 --frames 30000000  --env 'MiniGrid-MultiRoom-N7-S8-v0' --intrinsic-motivation 0.005 --im-type 'counts' --int-coef-type 'ngu' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 0  --use-gpu 1  --gpu-id 1&
nohup python3 -m scripts.train --model MN7S8_c_ngu_im0005_ent00005_2  --seed 2 --save-interval 10 --frames 30000000  --env 'MiniGrid-MultiRoom-N7-S8-v0' --intrinsic-motivation 0.005 --im-type 'counts' --int-coef-type 'ngu' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 0  --use-gpu 1  --gpu-id 1&
nohup python3 -m scripts.train --model MN7S8_c_ngu_im0005_ent00005_3  --seed 3 --save-interval 10 --frames 30000000  --env 'MiniGrid-MultiRoom-N7-S8-v0' --intrinsic-motivation 0.005 --im-type 'counts' --int-coef-type 'ngu' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 0  --use-gpu 1  --gpu-id 1&
