"""
Last modified date: 2023.02.23
Author: Ruicheng Wang
Description: execute shell script for grasp validation with multiple processes
"""

from multiprocessing import Pool
import os
import time
import argparse


def run_cmd(cmd):
    if cmd[-1] == "\n":
        cmd = cmd[:-1]
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        help="input shell script",
                        default="run.sh")
    parser.add_argument("-p",
                        "--process",
                        type=int,
                        help="num of process to run",
                        default=4)
    args = parser.parse_args()

    p = Pool(args.process)

    cmds = None

    with open(args.input, "r") as f:
        cmds = f.readlines()

    t1 = time.time()

    for i in cmds:
        if i[0] == '\n':
            continue
        p.apply_async(run_cmd, args=(i, ))

    p.close()
    p.join()

    t2 = time.time()

    print(f"Finished in {(t2-t1):.2f}s")
