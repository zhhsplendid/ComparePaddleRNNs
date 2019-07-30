from rnn import SimpleRnn, benchmark


def main():
    print("DynamicRNN benchmark")
    dynamic_rnn = SimpleRnn()
    dynamic_rnn.set_up_dynamic()
    benchmark(dynamic_rnn)


if __name__ == '__main__':
    main()
