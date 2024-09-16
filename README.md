# ReGentS: Real-World Safety-Critical Driving Scenario Generation Made Stable

The official repository of ReGentS. The work is accepted to [ECCV 2024 W-CODA workshop](https://coda-dataset.github.io/w-coda2024/ "ECCV 2024 Workshop on Multimodal Perception and Comprehension of Corner Cases in Autonomous Driving").

> Abstract: Machine learning based autonomous driving systems often face challenges with safety-critical scenarios that are rare in real-world data, hindering their large-scale deployment. While increasing real-world training data coverage could address this issue, it is costly and dangerous. This work explores generating safety-critical driving scenarios by modifying complex real-world regular scenarios through trajectory optimization. We propose ReGentS, which stabilizes generated trajectories and introduces heuristics to avoid obvious collisions and optimization problems. Our approach addresses unrealistic diverging trajectories and unavoidable collision scenarios that are not useful for training robust planner. We also extend the scenario generation framework to handle real-world data with up to 32 agents. Additionally, by using a differentiable simulator, our approach simplifies gradient descent-based optimization involving a simulator, paving the way for future advancements.

```
@misc{YinKZNC2024,
   title={ReGentS: Real-World Safety-Critical Driving Scenario Generation Made Stable}, 
   author={Yuan Yin and Pegah Khayatan and Éloi Zablocki and Alexandre Boulch and Matthieu Cord},
   year={2024},
   eprint={2409.07830},
   archivePrefix={arXiv},
   primaryClass={cs.LG},
   url={https://arxiv.org/abs/2409.07830}, 
}
```


## Usage

We are required not to provide the trained ego planner with WOMD. Follow these steps to use ReGentS with a reactive ego agent:
- Download the WOMD v1.1
- Preprocess the training data using `preprocess_data.py`. This will create a dataset that stores ego-centric observations and target points.
- Train the ego planner model with `train_agent.py`. The path to the downloaded WOMD training dataset can be specified in `config_aim_bev.yaml`
- Import the trained model into the AIM-BEV ego agent configuration in `config_scenario_opt.yaml`. The path to the downloaded WOMD evaluation dataset can be also specified there.
- Launch ReGentS using `generate_scenario.py`


## Notice

### Waymax

This software was made using the Waymax Licensed Materials, provided by Waymo LLC under the Waymax License Agreement for Non-Commercial Use, available at https://github.com/waymo-research/waymax/blob/main/LICENSE, and your access and use of the Waymax Licensed Materials are governed by the terms and conditions contained therein.

### KING

This software uses part of the code from the AIM-BEV model, available at https://github.com/autonomousvision/king, for the ego driving agent to ensure consistency.


