import sys

from midiplayer.cli import get_parser
from midiplayer import player

def main() -> None:
    parser = get_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    player.run(args)


if __name__ == "__main__":
    main()
