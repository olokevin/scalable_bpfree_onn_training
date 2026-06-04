"""
"""
from pyutils.general import AverageMeter


def test_averagemeter():
    acc = AverageMeter("Acc", ":.4f")
    acc.update(10, 1)
    acc.update(20, 1)
    acc.update(40, 1)
    print(acc)


if __name__ == "__main__":
    test_averagemeter()
