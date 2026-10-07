#!/usr/bin/env bash
# Build the SonicBoom-side dump tool against libsonicboom.so (the SHARED core).
# It links only the published Layer 2 library and includes only <sonicboom/...>
# public headers, mirroring how the tests are built.
set -euo pipefail

ROOT="${SONICBOOM_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LIB="$ROOT/build/core"
INC="$ROOT/core/include"

CXX="${CXX:-g++-13}"
CXXFLAGS="-std=c++23 -O2 -I$INC"
LDFLAGS="-L$LIB -Wl,-rpath,$LIB -lsonicboom"

"$CXX" $CXXFLAGS "$(dirname "$0")/sonicboom_dump.cpp" -o "$(dirname "$0")/sonicboom_dump" $LDFLAGS

echo "built $(dirname "$0")/sonicboom_dump"
