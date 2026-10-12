"""Count a sync log's lines per level and its most common errors: `python count_levels.py <log file>`."""

import sys
from collections import Counter


def main(path: str) -> None:
    levels, errors = Counter(), Counter()
    with open(path, encoding="utf-8") as log:
        for line in log:
            parts = line.split(maxsplit=3)
            if len(parts) == 4:
                levels[parts[1]] += 1
                if parts[1] == "ERROR":
                    errors[parts[3].strip()] += 1
    print("levels:", dict(levels))
    for message, count in errors.most_common(3):
        print(f"{count} x {message}")


if __name__ == "__main__":
    main(sys.argv[1])
