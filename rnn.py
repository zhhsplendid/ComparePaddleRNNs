import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
import time

from paddle.fluid import ParamAttr
from paddle.fluid.framework import Program, grad_var_name
from paddle.fluid.executor import Executor
from paddle.fluid.backward import append_backward

np.random.seed(123)

SEQ_LEN = 1000
BATCH_SIZE = 20
INPUT_DIM = 50

NUM_ITERATION = 10

DEFAULT_PLACE = core.CUDAPlace(0)


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

    def set_up_dynamic(self, x):
        self.set_up(False, x)

    def set_up_static(self, x):
        self.set_up(True, x)

    def set_up(self, static, x):
        self.main_program = Program()
        self.startup_program = Program()
        self.x = x
        self.scale = 1.0
        with fluid.program_guard(self.main_program, self.startup_program):
            self.output = self.net(static)
        lod = [[SEQ_LEN for _ in range(BATCH_SIZE)]]
        x_tensor = create_tensor(self.x,
                                 DEFAULT_PLACE) if static else create_tensor(
                                     self.x, DEFAULT_PLACE, lod)
        self.feed_map = {'x': x_tensor}
        self.grad_var_list = {self.main_program.global_block().var('x')}

        self.exe = Executor(DEFAULT_PLACE)
        self.exe.run(self.startup_program)

    def net(self, static):
        if static:
            x = layers.data(shape=[SEQ_LEN, BATCH_SIZE, INPUT_DIM],
                            dtype='float32',
                            name='x',
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
        else:
            x = layers.data(shape=[BATCH_SIZE * SEQ_LEN, INPUT_DIM],
                            dtype='float32',
                            name='x',
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
        #layers.Print(rnn())
        return layers.mean(rnn())

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


def benchmark(is_static):
    x = np.random.uniform(low=0.0,
                          high=1.0,
                          size=(SEQ_LEN, BATCH_SIZE,
                                INPUT_DIM)).astype("float32")
    rnn = SimpleRnn()
    if is_static:
        rnn.set_up_static(x)
    else:
        x = np.transpose(x, axes=(1, 0, 2)).reshape(
            (SEQ_LEN * BATCH_SIZE, INPUT_DIM))
        rnn.set_up_dynamic(x)

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
    benchmark(is_static=False)
    print("StaticRNN benchmark")
    benchmark(is_static=True)


if __name__ == '__main__':
    main()
