# ComparePaddleRNNs

Compare static and dynamic RNNs of PaddlePaddle

Usage:

Please install PaddlePaddle before running the scripts in this repo

```
sh run.sh
```

Current result (updated on July 31, 2019)


StaticRNN benchmark
Takes 1.363774 sec to forward 10 times,
Takes 2.570426 sec to backward 10 times,

DynamicRNN benchmark
Takes 1.557857 sec to forward 10 times,
Takes 32.451699 sec to backward 10 times,
