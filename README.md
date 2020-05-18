# Dataset 
There are totally 4 map corresponding to different dataset_id (0, ..., 3)
[Dataset](https://drive.google.com/drive/folders/1LdD9B1mpaMSgoy5itutOOqpuzR6DfZVz?usp=sharing) includes a train directory, which includes 4 datasets, each dataset has train_joint<dataset_id>.mat and train_lidar<dataset_id>.mat

# Run on dataset
```
python main.py --split_name train --dataset_id <0, 1 or 2, or 3>
```

# Generate figures 
To generate figures, run
```
python gen_figures.py --split_name <train or test>  --dataset_id <0, 1, or 2, 3> 
```

# Log files
Log file & images are all stored in "logs" repository

# Results

![map0](/results/0.jpg)

![map1](/results/1.jpg)

![map2](/results/2.jpg)

![map3](/results/3.jpg)
