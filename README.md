# VDAct
This repository contains code and data for "A Video-grounded Dialogue Dataset and Metric for Event-driven Activities"

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
3. please set the path of model output file (json format) and evaluation results directory in `run_vdeval.sh`
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
