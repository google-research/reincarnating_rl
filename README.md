
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ktlNni_vwFpFtCgUez-RHW0OdGc2U_Wv?usp=sharing) [![Website](https://img.shields.io/badge/www-Website-green)](https://agarwl.github.io/reincarnating_rl) [![Blog](https://img.shields.io/badge/b-Blog-blue)](https://ai.googleblog.com/2022/11/beyond-tabula-rasa-reincarnating.html)

# Reincarnating Reinforcement Learning: Reusing Prior Computation to Accelerate Progress

<p class="cover" align="center"> <img src="https://agarwl.github.io/images/research/RRL.gif" width="90%" /> </p>

This codebase provides the open source implementation using the
[Dopamine][dopamine] framework for running Atari experiments in
[Reincarnating RL][paper]. In this work, we leverage the
policy from an existing agent (e.g., DQN trained for 400M environment frames) to
reincarnate another deep Q-learning agent. Refer to
[agarwl.github.io/reincarnating_rl][project_page] for the project page.

*This release is a work-in-progress. More instructions to be added soon.*

## Downloading Teacher Checkpoints

The teacher checkpoints for pre-trained deep RL agents are in the public GCP
bucket `gs://rl_checkpoints` ([browser link][gcp_bucket]) which can be
downloaded using [`gsutil`][gsutil]. To install gsutil, follow the instructions
[here][gsutil_install].

After installing gsutil, run the command to download the final checkpoint and
Dopamine replay buffer for a DQN (Adam) agent trained for 400 million
environment frames on Atari 2600 games:

```
gsutil -m cp -R gs://rl_checkpoints/DQN_400 ./
```

To run the dataset only for a specific Atari game (*e.g.*, replace `GAME_NAME`
by `Breakout` to download the checkpoint for the game of Breakout), run the
command:

```
gsutil -m cp -R gs://rl_checkpoints/DQN_400/[GAME_NAME] ./
```

Note that the agents were trained using recommended training protocol on Atari
with sticky actions, *i.e.*, there is 25% chance at every time step that the
environment will execute the agent's previous action again, instead of the
agent's new action.

## Installation

Install `Dopamine` as a library following the
[instructions here](https://github.com/google/dopamine#installing-from-source).
Alternative, use the following command:

```
pip install git+https://github.com/google/dopamine.git
```

For using Atari environments, follow the instructions provided in
[Dopamine prerequisites](https://github.com/google/dopamine#prerequisites).

1.  Install the atari roms following the instructions from
    [atari-py](https://github.com/openai/atari-py#roms).
2.  `pip install ale-py` (we recommend using a
    [virtual environment](virtualenv)):
3.  `unzip $ROM_DIR/ROMS.zip -d $ROM_DIR && ale-import-roms $ROM_DIR/ROMS`
    (replace $ROM_DIR with the directory you extracted the ROMs to).

Once you have setup `Dopamine`, clone this repository:

```
git clone https://github.com/google-research/reincarnating_rl.git
```

### Running the code

The entry point for training policy to value reincarnating RL (PVRL) agents on
Atari 2600 games is
[reincarnating_rl/train.py](https://github.com/google-research/reincarnating_rl/reincarnating_rl/train.py).

To run any PVRL agent given a teacher agent, we need to first download the
teacher checkpoints to `$TEACHER_CKPT_DIR`. To do so, we download the
checkpoints of a DQN (Adam) trained for 400M frames on `Breakout`.

```
export TEACHER_CKPT_DIR="<Insert directory name here>"
mkdir -p $TEACHER_CKPT_DIR/Breakout
gsutil -m cp -R gs://rl_checkpoints/DQN_400/Breakout $TEACHER_CKPT_DIR
```

Assuming that you have cloned the [reincarnating_rl][repo] repository, run the
`QDaggerRainbow` agent using the following command:

```
cd reincarnating_rl
python -um reincarnating_rl.train \
  --agent qdagger_rainbow \
  --gin_files reincarnating_rl/configs/qdagger_rainbow.gin
  --base_dir /tmp/qdagger_rainbow \
  --teacher_checkpoint_dir $TEACHER_CKPT_DIR/Breakout/1 \
  --teacher_checkpoint_number 399
  --run_number=1 \
  --atari_roms_path=/tmp/atari_roms \
  --alsologtostderr
```

To use a `Impala CNN` architecture for the rainbow agent, pass the flag
`--gin_bindings @reincarnation_networks.ImpalaRainbowNetwork` to the above
command. More generally, since this code is based on Dopamine, it can be easily
configured using the [gin configuration](https://github.com/google/gin-config)
framework.

To run a quick experiment run for testing / debugging, you can use the following
command:

```
python -um reincarnating_rl.train \
  --agent qdagger_rainbow \
  --gin_files reincarnating_rl/configs/qdagger_rainbow.gin
  --base_dir /tmp/qdagger_rainbow \
  --teacher_checkpoint_dir $TEACHER_CKPT_DIR/Breakout/1 \
  --teacher_checkpoint_number 399 \
  --atari_roms_path=/tmp/atari_roms \
  --run_number=1 \
  --gin_bindings="Runner.evaluation_steps=10" \
  --gin_bindings="RunnerWithTeacher.num_pretraining_iterations=2" \
  --gin_bindings="RunnerWithTeacher.num_pretraining_steps=10" \
  --gin_bindings="JaxDQNAgent.min_replay_history = 64" \
  --alsologtostderr
```

[gsutil_install]: https://cloud.google.com/storage/docs/gsutil_install#install
[gsutil]: https://cloud.google.com/storage/docs/gsutil
[ale]: https://github.com/mgbellemare/Arcade-Learning-Environment
[gcp_bucket]: https://console.cloud.google.com/storage/browser/rl_checkpoints
[project_page]: https://agarwl.github.io/reincarnating_rl
[paper]: https://arxiv.org/pdf/2206.01626.pdf
[dopamine]: https://github.com/google/dopamine
[repo]: https://github.com/google-research/reincarnating_rl

### Citing

If you find this open source release useful, please reference in your paper:

> Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G.
> (2022). Reincarnating Reinforcement Learning: Reusing Prior Computation 
> to Accelerate Progress *arXiv preprint arXiv:2206.01626*.

```
@inproceedings{agarwal2022beyond,
  title={Reincarnating Reinforcement Learning: Reusing Prior Computation to Accelerate Progress},
  author={Agarwal, Rishabh and Schwarzer, Max and Castro, Pablo Samuel and Courville, Aaron and Bellemare, Marc G},
  booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
  year={2022}
}
```

*Disclaimer: This is not an official Google product.*
