# Banana Collector

We train an ```agent``` to navigate in a large square shaped space and collect yellow bananas while avoiding blue bananas. The agent interacts and receives feedback from ([Unity ML Agent](https://github.com/Unity-Technologies/ml-agents)) envionment using Python API. The environment is considered solved when the agent manages to collect 13 bananas on average over 100 consecutive episodes.

Trained agent collects bananas like a pro:

![trained_agent](outputs/trained_agent.gif)

Read more about training process and results in the [report](/Report.md). :monkey:

### Getting Started

To run the code, you need Python 3.6 environment with required dependencies installed.
1. Create environment

```
conda create --name bananaproject python=3.6
source activate bananaproject
```


2. Clone this repository and install requirements

```
git clone https://github.com/tomkommando/BananaProject.git
cd BananaProject
pip install -r requirements.txt
```

3. You may need to download a Udacity Banana Project environment. Pick a version that match your operating system.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the BananaProject GitHub repository folder, and unzip (or decompress) the file.
 
### Instructions

Follow the instructions in [this notebook](train_agent.ipynb) to train an agent or watch a trained agent playing!

