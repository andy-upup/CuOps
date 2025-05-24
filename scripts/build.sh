#!/bin/bash

ROOT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
echo "Root path: $ROOT_PATH"
BUILD_DIR="$ROOT_PATH/build"

if [ ! -d "${BUILD_DIR}" ]; then
  echo "Creating build directory at $BUILD_DIR"
  mkdir -p "$BUILD_DIR"
else
    echo "Build directory already exists at $BUILD_DIR"
fi

cd "$BUILD_DIR" || exit 1
echo "Building in $BUILD_DIR"

cmake -DCMAKE_BUILD_TYPE=Release \
      ..

make -j16