from rnn import SimpleRnn, DYNAMIC_RNN_MODE, benchmark


def main():
    print("DynamicRNN benchmark")
    benchmark(DYNAMIC_RNN_MODE)


if __name__ == '__main__':
    main()
