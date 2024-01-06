from rock.data import get_congressional_dataset
from rock.process import get_rock_input
from rock.rock_algorithm import RockAlgorithm

THETA = 0.8
K = 2
APPROX_FN = lambda x: (1 - x) / (1 + x)


def main():
    dataset = get_congressional_dataset()
    rock_input = get_rock_input(dataset=dataset)
    rock = RockAlgorithm(input_dataset=rock_input, theta=THETA, k=K, approx_fn=APPROX_FN)
    rock.run()


if __name__ == "__main__":
    main()
