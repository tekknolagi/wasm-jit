#!/bin/bash

set -eux -o pipefail

clang-format -i interp.cc
clang-tidy --fix interp.cc
