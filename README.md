# Pysarl框架介绍
Pysarl框架，全称Python Single Agent Reinforcement Learning，是基于pytorch和gymnasium实现的单智能体强化学习框架，其下包含多种算法及其变种，对接gymnasium最新接口，支持大部分gymnasium环境

## 快速开始
使用命令

```python 
    python src/main.py with alg=dqn env_args.gamename=CartPole-v1
```

## SACRED库和args全局配置变量
1. 项目使用sacred库管理实验和实验参数args，args变量是一个全局性的配置变量，主导了框架内所有组件的一切行为，args来自于有四个来源，优先级为default.yaml<alg.yaml<环境参数<命令行参数
2. 在实验开始前，程序自动读取config之下default.yaml和alg.yaml的参数，其中前者包含日志控制、模型加载和保存等基础参数，后者则包含所有算法的默认参数
3. 除了default.yaml和alg.yaml的参数以外，程序自动读取指定环境的yaml配置，sacred库自动处理命令行参数
