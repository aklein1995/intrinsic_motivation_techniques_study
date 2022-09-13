 #!/bin/bash

nohup python3 -m scripts.train --model KS3R3_r_ad_im005_ent00005_1 --seed 1 --save-interval 10 --frames 30000000  --env 'MiniGrid-KeyCorridorS3R3-v0' --intrinsic-motivation 0.05 --im-type 'ride' --int-coef-type 'adaptive' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 0  --use-gpu 1  --gpu-id 3&
nohup python3 -m scripts.train --model KS3R3_r_ad_im005_ent00005_2 --seed 2 --save-interval 10 --frames 30000000  --env 'MiniGrid-KeyCorridorS3R3-v0' --intrinsic-motivation 0.05 --im-type 'ride' --int-coef-type 'adaptive' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 0  --use-gpu 1  --gpu-id 3&
nohup python3 -m scripts.train --model KS3R3_r_ad_im005_ent00005_3 --seed 3 --save-interval 10 --frames 30000000  --env 'MiniGrid-KeyCorridorS3R3-v0' --intrinsic-motivation 0.05 --im-type 'ride' --int-coef-type 'adaptive' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 0  --use-gpu 1  --gpu-id 3&
