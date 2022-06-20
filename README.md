# Beyond Tabula Rasa: Reincarnating Reinforcement Learning

This codebase provides the open source implementation using the
[Dopamine][dopamine] framework for running Atari experiments in [Beyond Tabula Rasa: Reincarnating RL][paper].
In this work, we leverage the policy from an existing agent (e.g., DQN trained 
for 400M environment frames) to reincarnate another deep Q-learning agent. Refer
to [reincarnating-rl.github.io][project_page] for the project page.

[gsutil_install]: https://cloud.google.com/storage/docs/gsutil_install#install
[gsutil]: https://cloud.google.com/storage/docs/gsutil
[ale]: https://github.com/mgbellemare/Arcade-Learning-Environment
[gcp_bucket]: https://console.cloud.google.com/storage/browser/atari-replay-datasets
[project_page]: https://reincarnating-rl.github.io
[paper]: https://arxiv.org/pdf/2206.01626.pdf
[dopamine]: https://github.com/google/dopamine

### Citing
------
If you find this open source release useful, please reference in your paper:

> Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G. (2022).
> Beyond Tabula Rasa: Reincarnating Reinforcement Learning
> *arXiv preprint arXiv:2206.01626*.

    @article{agarwal2022beyond,
      title={Beyond Tabula Rasa: Reincarnating Reinforcement Learning},
      author={Agarwal, Rishabh and Schwarzer, Max and Castro, Pablo Samuel and Courville, Aaron and Bellemare, Marc G},
      journal={arXiv preprint arXiv:2206.01626},
      year={2022}
    }

*Disclaimer: This is not an official Google product.*