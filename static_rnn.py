from rnn import SimpleRnn, STATIC_RNN_MODE, benchmark


def main():
    print("StaticRNN benchmark")
    benchmark(STATIC_RNN_MODE)


if __name__ == '__main__':
    main()
