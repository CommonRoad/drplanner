# DrPlanner 🩺 Dataset

<code> <i>Please </i> <b>DONOT</b> <i> merge this branch to the other ones! Keep them confidential!</i></code>

This repository contains the dataset for the DrPlanner project. Currently, there are ~50 data points in the dataset. Each data point contains the following information:
- **input**: the heuristic function to be diagnosed (<mark>todo</mark>: the motion primitives need to be added)
- **output**:
  1. summary: diagnosis and prescription pairs
  2. improved_heuristic_function: the diagnosis of the heuristic function (<mark>todo</mark>: the repaired motion primitives need to be added)
- **jsons**: the desired input for finetuning the GPT/codeLlama, might be the other types of data for different variants of LLMs
- **raw**: raw file for you to better enter the data

The **name** of each file is obtained from the student submissions to the commonroad-search exercise. I will give you more students' submissions soon. Again, pls DONOT distribute them.