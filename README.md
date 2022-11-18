# SparseExpertFeedback

https://github.com/hiwonjoon/ICML2019-TREX
https://github.com/dsbrown1331/CoRL2019-DREX
https://github.com/dsbrown1331/bayesianrex

python LearnAtariReward.py --env_name pong --models_dir . --reward_model_path ./learned_models/srihari_test.params

python VisualizeAtariLearnedReward.py --env_name pong --models_dir . --reward_net_path ./learned_models/srihari_test.params --save_fig_dir ./viz

T-REX script is generating sample trajectories using a PPO agent. Then it is doing a pairwise comparison between trajectories to determine rank. Finally an agent is trained to discriminate between two trajectories (better or worse).