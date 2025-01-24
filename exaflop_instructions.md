1. Run the following in the folder you clone this repo to:

```
pip install -r requirements.txt
# for texture
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
```

2. Set $HF_HOME to a location in /workspace because the model is big (and set the network volume storage to a high number, like 50GB)
```
export HF_HOME=<path in workspace>
```

3. Navigate to ```/Hunyuan3D-2``` (the location you cloned the repo to) and run ```python exaflop_functions.py```

Note: the runpod image used is ```runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04```
