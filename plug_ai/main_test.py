import argparse



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_argument", type=str, default="This is the default value")
    args = parser.parse_args()
    return args


def main(args):
    print("Arguments : ", args)
    

if __name__ == '__main__':
    args = parse_args()

    main(args)
