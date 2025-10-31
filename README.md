# HPML-course-materials
Contains all course materials from the HPML group
Course environment: https://ondemand.snellius.surf.nl
- Login with scurXXX login
- Click on "Jupyter"
- Select "partition" -> gpu_course
- "Select environment module version" -> Course
- Memory: 16 (GB)
- CPU cores: 2
- GPUs: 1
- time: e.g. 1:30:00 (1h30m)


## Course Overview  
- Hardware (e.g. Tensor cores) and software features (e.g. low level libraries for deep learning) for accelerated deep learning
- Packed data formats
- Profiling PyTorch with TensorBoard
- Parallel computing for deep learning

## Profiling
```bash
python3 -m venv venv
source venv/bin/activate
pip install git+https://github.com/pytorch/kineto.git#subdirectory=tb_plugin

git clone --depth=1 https://github.com/SURF-ML/HPML-course-materials.git

tensorboard --logdir HPML-course-materials/Day2/notebooks/logs/
```


## Other courses and resources
- AI Guide by LUMI: https://github.com/Lumi-supercomputer/LUMI-AI-Guide
- LLMs on supercomputers: https://gitlab.tuwien.ac.at/vsc-public/training/LLMs-on-supercomputers 
