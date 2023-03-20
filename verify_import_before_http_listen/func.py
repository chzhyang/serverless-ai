import os
import fakemodule
from parliament import Context

print("func static code")
fakemodule.myfunc1()


def inference():
    print("func-inference-function ")


def main(context: Context):
    print("func-main")
    inference()
    fakemodule.myfunc2()
