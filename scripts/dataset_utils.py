#!/usr/bin/env python

import re

# =======
# - KTH -
# =======


def read_sequences_file(filepath):
    """Parses the KTH sequences file
    and returns the relevant information as a dictionary.
    """
    info = {}
    pattern = re.compile('([a-z0-9_]+)\s+frames\s+(\d+)-(\d+), (\d+)-(\d+), (\d+)-(\d+), (\d+)-(\d+)')
    for line in open(filepath):
        for match in re.finditer(pattern, line):
            g = match.groups()
            info[g[0] + '_uncomp.avi'] = (
                (int(g[1]) - 1, int(g[2]) - 1),
                (int(g[3]) - 1, int(g[4]) - 1),
                (int(g[5]) - 1, int(g[6]) - 1),
                (int(g[7]) - 1, int(g[8]) - 1),
            )  # {title: (start0, end0, start1, end1, start2, end2, start3, end3)}
    return info
