from rnn import SimpleRnn, benchmark


def main():
    print("StaticRNN benchmark")
    static_rnn = SimpleRnn()
    static_rnn.set_up_static()
    benchmark(static_rnn)


if __name__ == '__main__':
    main()
