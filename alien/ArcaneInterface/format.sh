#!/bin/bash

find ./ -type f \( -iname \*.h -o -iname \*.cc -o -iname \*.cpp \) -exec clang-format -i -style=file {} \;
