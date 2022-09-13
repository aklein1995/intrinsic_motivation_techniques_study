
# train
python3 -m scripts.train --model TEST --save-interval 10 --frames 10000000 --env 'MiniGrid-MultiRoom-N7-S4-v0' --intrinsic-motivation 0.1 --separated-networks 0 --im-type rnd --use-episodic-counts 0 --procs 1 --nsteps 10
python3 -m scripts.train --model TEST --save-interval 10 --frames 10000000 --env 'MiniGrid-MultiRoom-N7-S4-v0' --intrinsic-motivation 0.1 --im-type rnd --use-episodic-counts 0 --procs 1 --nsteps 10 --int-coef-type parametric
python3 -m scripts.train --model TEST --save-interval 10 --frames 10000000 --env 'MiniGrid-Empty-5x5-v0' --intrinsic-motivation 0.005 --im-type counts --use-episodic-counts 0 --procs 16 --nsteps 10
python3 -m scripts.train --model TEST --save-interval 10 --frames 10000000 --env 'MiniGrid-Empty-5x5-v0' --intrinsic-motivation 0.005 --im-type counts --use-episodic-counts 0 --procs 16 --nsteps 10


# visualization
python3 -m scripts.visualize.py --env 'MiniGrid-MultiRoom-N7-S5-v0' --model MultiRoomN7S4/Counts/MN4S7_im0005_ent001/ --separated-networks 1


python3 -m scripts.visualize.py --env 'MiniGrid-MultiRoom-N7-S4-v0' --model MultiRoomN7S4/ICM/MN4S7_im01_ent00001/ --separated-networks 0

python3 -m scripts.visualize.py --env 'MiniGrid-MultiRoom-N7-S6-v0' --model MultiRoomN3S8/MN3S8_im0001_ent0001/ --separated-networks 1



python3 -m scripts.visualize.py --env 'MiniGrid-ObstructedMaze-2Dlh-v0' --model O2Dlh_r_im005_ent00005_1/ --separated-networks 0
