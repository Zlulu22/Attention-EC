python necsa_td3.py --task Hopper-v3  --epoch 400 --step 3 --grid_num 10 --epsilon 0.2 --mode state_action
pkill -f Hopper
python necsa_td3.py --task Hopper-v3  --epoch 400 --step 3 --grid_num 10 --epsilon 0.2 --mode state_action
pkill -f Hopper
python necsa_td3.py --task Hopper-v3  --epoch 400 --step 3 --grid_num 10 --epsilon 0.2 --mode state_action
pkill -f Hopper

python necsa_td3.py --task Swimmer-v3 --epoch 200 --step 3 --grid_num 10 --epsilon 0.15 --state_min -6 --state_max 6 --mode state_action
pkill -f Swimmer
python necsa_td3.py --task Swimmer-v3 --epoch 200 --step 3 --grid_num 10 --epsilon 0.15 --state_min -6 --state_max 6 --mode state_action
pkill -f Swimmer
python necsa_td3.py --task Swimmer-v3 --epoch 200 --step 3 --grid_num 10 --epsilon 0.15 --state_min -6 --state_max 6 --mode state_action
pkill -f Swimmer


python necsa_td3.py --task Ant-v3  --epoch 1000 --step 3 --grid_num 5 --state_dim 24 --state_min -6 --state_max 6 --epsilon 0.2 --mode state_action --reduction
pkill -f Ant
python necsa_td3.py --task Ant-v3  --epoch 1000 --step 3 --grid_num 5 --state_dim 24 --state_min -6 --state_max 6 --epsilon 0.2 --mode state_action --reduction
pkill -f Ant
python necsa_td3.py --task Ant-v3  --epoch 1000 --step 3 --grid_num 5 --state_dim 24 --state_min -6 --state_max 6 --epsilon 0.2 --mode state_action --reduction
pkill -f Ant

python necsa_td3.py --task HalfCheetah-v3 --epoch 1000 --step 3 --grid_num 5 --state_min -6 --state_max 6 --epsilon 0.2 --mode state_action
pkill -f HalfCheetah
python necsa_td3.py --task HalfCheetah-v3 --epoch 1000 --step 3 --grid_num 5 --state_min -6 --state_max 6 --epsilon 0.2 --mode state_action
pkill -f HalfCheetah
python necsa_td3.py --task HalfCheetah-v3 --epoch 1000 --step 3 --grid_num 5 --state_min -6 --state_max 6 --epsilon 0.2 --mode state_action
pkill -f HalfCheetah

python necsa_td3.py --task Humanoid-v3  --epoch 1000 --step 3 --grid_num 5 --state_dim 24 --state_min -6 --state_max 6 --epsilon 0.2 --mode state_action --reduction
pkill -f Humanoid
python necsa_td3.py --task Humanoid-v3  --epoch 1000 --step 3 --grid_num 5 --state_dim 24 --state_min -6 --state_max 6 --epsilon 0.2 --mode state_action --reduction
pkill -f Humanoid
python necsa_td3.py --task Humanoid-v3  --epoch 1000 --step 3 --grid_num 5 --state_dim 24 --state_min -6 --state_max 6 --epsilon 0.2 --mode state_action --reduction
pkill -f Humanoid


# python necsa_td3.py --task Walker2d-v3 --epoch 400 --step 3 --grid_num 5 --epsilon 0.2 --mode state_action
# pkill -f Walker2d
# python necsa_td3.py --task Walker2d-v3 --epoch 400 --step 3 --grid_num 5 --epsilon 0.2 --mode state_action
# pkill -f Walker2d
# python necsa_td3.py --task Walker2d-v3 --epoch 400 --step 3 --grid_num 5 --epsilon 0.2 --mode state_action
# pkill -f Walker2d