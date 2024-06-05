# pneumonitis_prediction

## Environment setup

- Step. 1
  
  Visit the link `https://services.northwestern.edu/TDClient/30/Portal/KB/ArticleDet?ID=1112`, and follow the instructions under `Pytorch` in `Anaconda` of `What GPU software is available on QUEST?` to create a conda environment with Pytorch installed.

- Step. 2
  
  ```
  conda activate pytorch-1.11-py38
  pip install -r ./requirements.txt
  ```
  
- Step. 3

  replace the environment path of the two `.sh` files in the code folder (run-info.sh, run-train.sh) with your own path

## Files Location

- code (demo is not on server): /home/phv0465/workspace/project-deliverable/code
- data: lung (/projects/p32050/lung); dose (/projects/p32050/dose); label (/projects/p32050/all_stats.csv)

## Train Model

On Northwestern Quest server

```
cd code
sbatch ./run-train.sh
```

## FPS/#Params/MACs Measurement

On Northwestern Quest server

```
cd code
sbatch ./run-info.sh
```

## Notebook

Please use `notebook.ipynb` in code folder for prediction and visualization.

## Demo

On a server with GPU in Northwestern. Have access to medical data.

```
cd demo
streamlit run ./demo.py
```

access the link shown in terminal for web interface.
