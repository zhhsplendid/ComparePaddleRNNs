from rnn import SimpleRnn, benchmark


def main():
    print("StaticRNN benchmark")
    benchmark(is_static=True)


if __name__ == '__main__':
    main()
