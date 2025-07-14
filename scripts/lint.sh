#!/bin/bash
find src include tests -name '*.cpp' -o -name '*.cu' -o -name '*.h' | xargs clang-format -i
