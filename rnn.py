import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
import time

from paddle.fluid import profiler
from paddle.fluid import ParamAttr
from paddle.fluid.framework import Program, grad_var_name
from paddle.fluid.executor import Executor
from paddle.fluid.backward import append_backward

np.random.seed(123)

SEQ_LEN = 1000
BATCH_SIZE = 20
INPUT_DIM = 50

NUM_ITERATION = 100

DEFAULT_PLACE = core.CUDAPlace(0)

STATIC_RNN_MODE = "static"
DYNAMIC_RNN_MODE = "dynamic"
FOR_RNN_MODE = "for"


def create_tensor(np_data, place, lod=None):
    tensor = core.LoDTensor()
    tensor.set(np_data, place)
    if lod:
        tensor.set_recursive_sequence_lengths(lod)
    return tensor


class SimpleRnn:
    '''
    Test RNNOp
    equation:
        h_t = ( x_t + h_{t-1} ) / scale
    vars:
        - x
    memories:
        - h
    outputs:
        - h
  '''

    def set_up_dynamic(self):
        self.set_up(DYNAMIC_RNN_MODE, x)

    def set_up_static(self):
        self.set_up(STATIC_RNN_MODE, x)

    def set_up_for(self):
        self.set_up(FOR_RNN_MODE, x)

    def set_up(self, rnn_mode, x):
        self.main_program = Program()
        self.startup_program = Program()
        self.x = x
        self.scale = 2.0
        with fluid.program_guard(self.main_program, self.startup_program):
            if rnn_mode == STATIC_RNN_MODE:
                self.output = self.static_rnn_net()
            elif rnn_mode == DYNAMIC_RNN_MODE:
                self.output = self.dynamic_rnn_net()
            else:
                self.output = self.for_rnn_net()
        lod = [[SEQ_LEN for _ in range(BATCH_SIZE)]]
        x_tensor = create_tensor(
            self.x, DEFAULT_PLACE,
            lod) if rnn_mode == DYNAMIC_RNN_MODE else create_tensor(
                self.x, DEFAULT_PLACE)
        self.feed_map = {'x': x_tensor}
        self.grad_var_list = {self.main_program.global_block().var('x')}

        self.exe = Executor(DEFAULT_PLACE)
        self.exe.run(self.startup_program)

    def static_rnn_net(self):
        x = layers.data(shape=[SEQ_LEN, BATCH_SIZE, INPUT_DIM],
                        dtype="float32",
                        name="x",
                        append_batch_size=False)
        x.stop_gradient = False
        rnn = layers.StaticRNN()
        with rnn.step():
            x_t = rnn.step_input(x)
            h_pre = rnn.memory(shape=[-1, INPUT_DIM], batch_ref=x_t)
            h = layers.scale(x=layers.elementwise_add(x=h_pre, y=x_t),
                             scale=self.scale)
            rnn.update_memory(h_pre, h)
            rnn.output(h)
        return layers.mean(rnn())

    def dynamic_rnn_net(self):
        x = layers.data(shape=[BATCH_SIZE * SEQ_LEN, INPUT_DIM],
                        dtype="float32",
                        name="x",
                        append_batch_size=False)
        x.stop_gradient = False
        rnn = layers.DynamicRNN()
        with rnn.block():
            x_t = rnn.step_input(x)
            h_pre = rnn.memory(shape=[INPUT_DIM])
            h = layers.scale(x=layers.elementwise_add(x=h_pre, y=x_t),
                             scale=self.scale)
            rnn.update_memory(h_pre, h)
            rnn.output(h)
        return layers.mean(rnn())

    def for_rnn_net(self):
        x = layers.data(shape=[BATCH_SIZE, SEQ_LEN, INPUT_DIM],
                        dtype="float32",
                        name="x",
                        append_batch_size=False)
        split_x = layers.split(x, num_or_sections=SEQ_LEN, dim=1)

        h_pre = fluid.layers.zeros(shape=[BATCH_SIZE, 1, INPUT_DIM],
                                   dtype="float32")

        for i in range(SEQ_LEN):
            x_t = split_x[i]
            h = layers.scale(x=layers.elementwise_add(x=h_pre, y=x_t),
                             scale=self.scale)
            h_pre = h
        return layers.mean(h_pre)

    def forward(self):
        out = self.exe.run(self.main_program,
                           feed=self.feed_map,
                           fetch_list=[self.output])
        return out[0]

    def prepare_backward(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            append_backward(self.output)

    def backward(self):
        out = self.exe.run(self.main_program,
                           feed=self.feed_map,
                           fetch_list=self.grad_var_list,
                           return_numpy=False)
        return out[0]


def benchmark(mode):
    x = np.random.uniform(low=0.0,
                          high=1.0,
                          size=(BATCH_SIZE, SEQ_LEN,
                                INPUT_DIM)).astype("float32")
    if mode == DYNAMIC_RNN_MODE:
        x = x.reshape(SEQ_LEN * BATCH_SIZE, INPUT_DIM)
    elif mode == STATIC_RNN_MODE:
        x = np.transpose(x, axes=(1, 0, 2))

    rnn = SimpleRnn()
    rnn.set_up(mode, x)

    start = time.time()
    for i in range(NUM_ITERATION):
        rnn.forward()
    end = time.time()
    print("Takes %f sec to forward %d times" % (end - start, NUM_ITERATION))

    rnn.prepare_backward()
    start = time.time()
    for i in range(NUM_ITERATION):
        rnn.backward()
    end = time.time()
    print("Takes %f sec to backward %d times" % (end - start, NUM_ITERATION))


def main():
    print("DynamicRNN benchmark")
    benchmark(mode=DYNAMIC_RNN_MODE)
    print("StaticRNN benchmark")
    benchmark(mode=STATIC_RNN_MODE)
    print("ForRNN benchmark")
    benchmark(mode=FOR_RNN_MODE)


if __name__ == '__main__':
    main()
