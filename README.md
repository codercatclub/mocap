# Video to character animation

## How to use on Windows

This required Python 3.9 and pip

```bash
# Create virtual enviroment and activate it
python -m venv ./env
./env/Scripts/activate

# Install dependencies
pip install -r ./requirements.txt

# Run pose estimation on test video
python ./pose.py ./videos/girl_dance.mp4 -v -o ./dance
```

Now open `mocap.hip` and select `./dance/pose/` folder on `pose_reader` node to map it to included character.