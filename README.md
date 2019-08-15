# ComparePaddleRNNs

Compare static and dynamic RNNs of PaddlePaddle, also a RNN implemented by "for" loop

## Usage:

Please install PaddlePaddle before running the scripts in this repo

```
sh run.sh
```

## Current result (updated on July 31, 2019)

StaticRNN benchmark

Takes 10.030948 sec to forward 100 times

Takes 21.970847 sec to backward 100 times

DynamicRNN benchmark

Takes 15.937193 sec to forward 100 times

Takes 310.427649 sec to backward 100 times

ForRNN benchmark

Takes 15.545057 sec to forward 100 times

Takes 32.583921 sec to backward 100 times

