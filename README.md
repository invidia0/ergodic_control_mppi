# Policy-Gradient Flow Matching for Real-Time Ergodic Control

This repository contains an implementation of the Ergodic Control using Model Predictive Path Integral (MPPI) method. The code is designed to facilitate research and experimentation in the field of ergodic control for robotic systems.

## Installation

This repository relies on uv package manager. You can install 
uv using the following command:

```bash
sudo apt-get install curl  # if curl is not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
```
After installing uv, you can clone this repository and install the required dependencies using:

```bash
uv sync
```

>[!NOTE]
> Make sure to select the correct Python venv in your IDE after installing the dependencies.

### No uv ?
If you do not wish to use uv, you can manually install the required dependencies using pip. First, create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
pip install .
```