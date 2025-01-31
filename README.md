# VDAct
This repository contains code and data for "[A Video-grounded Dialogue Dataset and Metric for Event-driven Activities](https://arxiv.org/abs/2501.18324)"

## Overview
VDAct is a dataset for Video-Grounded Dialogue on Event-driven Activities.
VDAct includes longer and more complex video sequences that depict a variety of event-driven activities that require advanced contextual understanding for accurate response generation.
The dataset comprises 3,000 dialogues with over 30,000 question-and-answer pairs, derived from 1,000 videos with diverse activity scenarios.
![Overview](https://github.com/aistairc/VDAct/blob/main/docs/dataset.png)

## Dataset
Dialogue data files: [dialgoue](https://github.com/aistairc/VDAct/blob/main/data/dialogue)

Knowledge graph data files: [kg](https://github.com/aistairc/VDAct/blob/main/data/kg)

Video files: please download from [here](https://kgrc4si.home.kg/Movie/)

## Open VLM inference
Please see [code](https://github.com/aistairc/VDAct/blob/main/code)

## VDEval
We propose a new metric for the visual dialogue task, VDEval: a session-based context evaluation metric that integrates dialogue session history and video content summaries extracted from the supplementary KGs of VDAct.

<img src="https://github.com/aistairc/VDAct/blob/main/docs/eval-visual.png" width="40%">

### Usage
1. `cd VDEval`
2. Set an API key `export OPENAI_API_KEY=<your key>`
3. Set the path of model output file (json format) and evaluation results directory in `run_vdeval.sh`
4. Run VDEval
```
sh run_vdeval.sh
```
5. Calculate score
```
python calc_score.py --model_output_file MODEL_OUTPUT_JSON --eval_output_file EVAL_OUTPUT_JSONL
```

## Citation
To appear soon

## Acknowledgment
This paper is based on results obtained from: (1) a project, Programs for Bridging the gap between R&D and the IDeal society (society 5.0) and Generating Economic and social value (BRIDGE)/Practical Global Research in the AI × Robotics Services, implemented by the Cabinet Office, Government of Japan, and (2) a project, JPNP20006, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).

In addition, we would like to thank Susan Holm, our knowledge engineer, for her insightful feedback during the data collection process.
