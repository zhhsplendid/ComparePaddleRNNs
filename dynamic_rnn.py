from rnn import SimpleRnn, benchmark


def main():
    print("DynamicRNN benchmark")
    benchmark(is_static=False)


if __name__ == '__main__':
    main()
