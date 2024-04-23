# LactChain: Language Action Chain Reinforcement Learning

This repo serves as a template for coding out a Reinforcement Learning (RL) system. This system is meant to be a multi-purpose system with multiple possible applications.

## Build Environment: 
To build an environment for LactChain, follow the instructions (this is for mac): 

```
# create and activate environment 
conda create -n lactchain python=3.11 -y
conda activate lactchain
```

pip install packages 
```
# install packages via pyproject.toml file
pip install -e .
```

pip install llama-cpp-python for arm64
```
# this is for m2/m1 chip macs
# ref link: https://github.com/abetlen/llama-cpp-python/issues/847#issuecomment-1869266853

CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DLLAMA_METAL=on" pip install --upgrade --verbose --force-reinstall --no-cache-dir llama-cpp-python
```


## TODOs:

Add things needed to enforce structure of any subsequent code
1. Make generic baseclass for actor (policy) network
2. Make generic baseclass for critic (value) network
3. Make generic baseclass for reward function
4. Finish thinking about generic lactchain baseclass. Yes, it is state-->action-->state, but what is action? Does action involve taking in a fluid prompt? A prompt menu? What?
5. Write unit tests

Build out specific use cases
1. Draw schematic of simple use case
2. Add plausibly useful language action chains using lactchain class
3. Add code extractor and other functions in state class
4. Add other extractors to lactchains if you need to pull certain things (like code) from gpt4 responses
5. Define example format for textblock in state class
6. Define Policy and Value Function networks
7. Define Actor-Critic teaching moments (TD learning? Whatever it's called)



