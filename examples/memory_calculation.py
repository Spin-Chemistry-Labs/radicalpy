#! /usr/bin/env python

import math


def main():

    def convert_size(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

    num_spins = 16
    hilbert = []
    liouville = []

    for N in range(num_spins):
        vec_size = 2**N
        vec_bytes = (vec_size * 16) ** 2
        hilbert.append((vec_size * 16) ** 2)
        print(f"Hilbert space: N={N:2} -> {convert_size(vec_bytes):15}")

    for N in range(num_spins):
        vec_size = (2**N) ** 2
        vec_bytes = (vec_size * 16) ** 2
        liouville.append((vec_size * 16) ** 2)
        print(f"Liouville space: N={N:2} -> {convert_size(vec_bytes):15}")


if __name__ == "__main__":
    main()
