# RL Atari Breakout
This projects aims to create a Deep Q-Network agent able to play Atari 2600 game Breakout.

The model is a close replication to the one proposed in 2015 DeepMind's *"Human Level Control Through Deep Reinforcement Learning"* publication [[1]](https://www.deepmind.com/publications/human-level-control-through-deep-reinforcement-learning). The model building was algo guided by the "Hands-On Machine Learning with Scikit-Learn and TensorFlow"* book [[2]](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurélien Géron, chapter *"Reinforcement Learning"*.

## How to run?
```bash
# this project uses poetry as virtual environment and package manager, it's pretty good!
# check it out and install it from here: https://python-poetry.org/docs/

# install all the dependencies with poetry package manager
# run poetry install and it will download/resolve all the required dependencies
>> poetry install

# Run setup.sh, this will install all utility tools and resources required.
# For the most part it will download roms and import them with ale-py
>> ./setup.sh

# If all of this has resolved in no errors, you're good to go!
# Simply run the model.py and watch the DQN agent train :)
>> python model.py
```
You can edit all the parameters such as train iterations or replay buffer sizes inside ```model.py```.  
Once the program finishes, there should be an **.mp4** file with the trained agent playing one round of breakout!


<p align="center">
  <img src="https://user-images.githubusercontent.com/24988290/162645967-0d92a2cc-00ba-4f0c-a91a-ee3ad2c7dca0.gif" width="300" height="400" />
</p>




## References
[1] Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236

[2] Aurélien Géron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition. https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/
