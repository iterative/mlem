import sys

import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", *sys.argv[1:]]))
