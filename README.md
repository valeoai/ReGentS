# ReGentS: Real-World Safety-Critical Driving Scenario Generation Made Stable

The official repository of ReGentS. The work is accepted to [ECCV 2024 W-CODA workshop](https://coda-dataset.github.io/w-coda2024/ "ECCV 2024 Workshop on Multimodal Perception and Comprehension of Corner Cases in Autonomous Driving").

> Abstract: Machine learning based autonomous driving systems often face challenges with safety-critical scenarios that are rare in real-world data, hindering their large-scale deployment. While increasing real-world training data coverage could address this issue, it is costly and dangerous. This work explores generating safety-critical driving scenarios by modifying complex real-world regular scenarios through trajectory optimization. We propose ReGentS, which stabilizes generated trajectories and introduces heuristics to avoid obvious collisions and optimization problems. Our approach addresses unrealistic diverging trajectories and unavoidable collision scenarios that are not useful for training robust planner. We also extend the scenario generation framework to handle real-world data with up to 32 agents. Additionally, by using a differentiable simulator, our approach simplifies gradient descent-based optimization involving a simulator, paving the way for future advancements.

```
@inproceedings{yin2024regents,
   title={{ReGentS}: Real-World Safety-Critical Driving Scenario Generation Made Stable}, 
   author={Yuan Yin and Pegah Khayatan and \'Eloi Zablocki and Alexandre Boulch and Matthieu Cord},
   booktitle={ECCV 2024 Workshop on Multimodal Perception and Comprehension of Corner Cases in Autonomous Driving},
   year={2024},
   url={https://openreview.net/forum?id=dJqcdUgEdw}
}
```


## Usage

We are required not to provide the trained ego planner with WOMD. Follow these steps to use ReGentS with a reactive ego agent:
- Download the WOMD v1.1
- Preprocess the training data using `preprocess_data.py`. This will create a dataset that stores ego-centric observations and target points.
- Train the ego planner model with `train_agent.py`. The path to the downloaded WOMD training dataset can be specified in `config_aim_bev.yaml`
- Import the trained model into the AIM-BEV ego agent configuration in `config_scenario_opt.yaml`. The path to the downloaded WOMD evaluation dataset can be also specified there.
- Launch ReGentS using `generate_scenario.py`
- 
### Requirement installation

The code is compatible with Python <=3.11 on Linux servers. 
Follow the instructions below to create your conda environment for ReGentS.

```
conda create --name regents python=3.11
conda activate regents
pip install -U -r requirements.txt
```

## License

This code is distributed under MPL 2.0 License (see [LICENSE](/LICENSE)). Please note that the portions of code developed with Waymax cannot be used for commercial purposes.

## Notice

See [NOTICE](/NOTICE).


