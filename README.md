# Image Retrieval Testbed for Indoor Robot Global Localization
--- 
This repository is for testing image retrieval methods for robot global localization.  
The test is focused on indoor environments from [Matterport3D Dataset](https://niessner.github.io/Matterport/).  
[Hierarchical Localization toolbox (hloc)](https://github.com/cvg/Hierarchical-Localization) is mainly tested for the image retrieval method.  

Following features are implemented.
   - Extract image retrieval database using [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
   - Test hloc toolbox in various indoor environments

If you are interested in more information, please visit our [blog]().


## Installation

### Prerequisite
Three 3rd party libraries are required.
- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)
- [Hierarchical Localization toolbox](https://github.com/cvg/Hierarchical-Localization)
- [Matterport3D Dataset](https://niessner.github.io/Matterport/)

#### Habitat-Sim
[Offical installation guide for Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
We tested with `habitat-sim` version `v0.2.1` and `v0.2.2`, but higher version should work.

```bash
# Make conda env
conda create -n habitat python=3.9 cmake=3.14.0
# Install with pybullet & GUI
conda install habitat-sim withbullet -c conda-forge -c aihabitat
# (Optional) When you get stuck in "Solving environment", put this command and try again
conda config --set channel_priority flexible
```

#### Habitat-Lab
[Offical installation guide for Habitat-Lab](https://github.com/facebookresearch/habitat-lab)  
We tested with `habitat-lab` version `v0.2.1` and `v0.2.2`, but higher version should work.

```bash
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
```

#### Hierarchical Localization toolbox
[Offical installation guide for Hierarchical Localization toolbox](https://github.com/cvg/Hierarchical-Localization)  
Only difference is that we used `develop` flag package installation to import 3rd party in `hloc`.  
We tested with `hloc` version `v1.3`.

```bash
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/

pip install -r requirements.txt
git submodule update --init --recursive
python setup.py develop  # To import 3rd party in `hloc`
```

#### Matterport3D Dataset
[Matterport3D dataset](https://niessner.github.io/Matterport/) is required for test in various indoor environments.  
To get this dataset, please visit [here](https://niessner.github.io/Matterport/) and submit Terms of Use agreement.  
You will get `download_mp.py` file after you submit the form.

```bash
# Download dataset. Need python 2.7. Dataset is about 15GB.
python2 download_mp.py --task habitat -o /your/path/to/download

# Make dataset directory.
# If you don't want to store dataset in this repo directory, fix SCENE_DIRECTORY in config file
cd ()  # clone of this repository
mkdir Matterport3D
cd Matterport3D

# Unzip 
unzip /your/path/to/download/v1/tasks/mp3d_habitat.zip -d ./
```


## Set Up
```bash
git clone git@github.com:kc-ml2/()
cd ()
pip install -r requirements.txt

# Test installation. This needs Matterport3D dataset below
python run_sim.py
```


## Pipeline

### Step 1. Generate top-down grid map image
```bash
# Because this step generates maps of all spaces(scenes), it takes quite a while
# We've already uploaded the result in ./data/, so you can skip this
python generate_grid_map.py
```

### Step 2. Generate database from grid map
```bash
# This step generates graph map, and gathers RGB observations assigned to each node
# It will occupy about 18GB of disk memory
python generate_map_observation.py
```

### Step 3. Run "Hierarchical Localization toolbox"
```bash
# This step extracts NetVLAD, Superpoint features and matching result
# It will occupy about 32GB of disk memory
python generate_hloc_feature.py
```

### Step 4. Run test on all scenes(spaces)
```bash
# Run global localization (retrieval) with graph map (database) and samples
# It iterates all scenes in scene_list_{setting}.txt file
python run_retrieval_test.py
# For visualization of each result and false case observation
python run_retrieval_test.py --visulaize
```


## Supported Methods

### Method 1. Hierarchical Localization (NetVLAD + Superpoint)
- This method is from [hloc toolbox](https://github.com/cvg/Hierarchical-Localization)
- See `config/concat_fourview_69FOV_HD.py` for example
- NetVLAD pre-trained weight is from [hloc toolbox](https://github.com/cvg/Hierarchical-Localization)
- Superpoint pre-trained weight for is from [SuperGlue by Magic Leap](https://github.com/magicleap/SuperGluePretrainedNetwork/tree/ddcf11f42e7e0732a0c4607648f9448ea8d73590)

### Method 2. Hierarchical Localization (NetVLAD only)
- This method is from [hloc toolbox](https://github.com/cvg/Hierarchical-Localization)
- See `config/concat_fourview_69FOV_NetVLAD_only.py` for example
- NetVLAD pre-trained weight is from [hloc toolbox](https://github.com/cvg/Hierarchical-Localization)

### Method 3. Brute-force matching (ORB)
- Brute-force matching ORB descriptors from two images. Top 30 matches are used for getting score
- It does not use DBOW for better accuracy
- See `config/concat_fourview_69FOV_HD_ORB.py` for example


## Result

For more information about metric, please visit our [blog]().  
Hierarchical localization (NetVLAD + Superpoint) performs best for image retrieval in Habitat-Sim.  
It is also superior in our real-world test scenario.

|Environment|Method|Accuracy|Distance [m] (std)|
|---|---|---|---|
|simulator|ORB (brute force match)|0.930|0.338 (0.574)|
|simulator|NetVLAD|0.967|0.234 (0.383)|
|simulator|NetVLAD + Superpoint|<span style="color: red">0.982</span>|<span style="color: red">0.174</span> (0.289)|
|real world|ORB (brute force match)|0.799|1.373 (3.714)|
|real world|NetVLAD|0.839|0.596 (0.838)|
|real world|NetVLAD + Superpoint|<span style="color: red">0.892</span>|<span style="color: red">0.465</span> (0.766)|


## Graph Map Generation


## Supported Observation Data Structure


## Test with Your Own Data




## Code Formatting
- Code formatting tool: `black`, `isort`
- Static code analysis tool: `pytest`, `flake8`, `pylint`

You can simply run formatting and static analysis with the commands below.

```bash
# Install black, isort, flake8, pylint and pytest.
# You can skip this if you already run Set Up above.
pip install -r requirements.txt

# Formatting
make format

# Static code analysis
make test
```


## License
This repository is MIT licensed.