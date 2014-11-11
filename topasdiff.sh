#!/usr/bin/env bash

# PATH="/Library/Frameworks/Python.framework/Versions/2.7/bin:${PATH}"
# export PATH

. "/Users/Smeets/cctbx/cctbx_build/setpaths.sh"

LIBTBX_BUILD="/Users/Smeets/cctbx/cctbx_build"
if [ -n "$PYTHONPATH" ]; then
  PYTHONPATH="$LIBTBX_BUILD/../cctbx_sources:$LIBTBX_BUILD/../cctbx_sources/clipper_adaptbx:$LIBTBX_BUILD/../cctbx_sources/docutils:$LIBTBX_BUILD/../cctbx_sources/boost_adaptbx:$LIBTBX_BUILD/../cctbx_sources/libtbx/pythonpath:$LIBTBX_BUILD/lib:$PYTHONPATH"
  export PYTHONPATH
else
  PYTHONPATH="$LIBTBX_BUILD/../cctbx_sources:$LIBTBX_BUILD/../cctbx_sources/clipper_adaptbx:$LIBTBX_BUILD/../cctbx_sources/docutils:$LIBTBX_BUILD/../cctbx_sources/boost_adaptbx:$LIBTBX_BUILD/../cctbx_sources/libtbx/pythonpath:$LIBTBX_BUILD/lib"
  export PYTHONPATH
fi
if [ -n "$DYLD_LIBRARY_PATH" ]; then
  DYLD_LIBRARY_PATH="$LIBTBX_BUILD/lib:$LIBTBX_BUILD/base/lib:$DYLD_LIBRARY_PATH"
  export DYLD_LIBRARY_PATH
else
  DYLD_LIBRARY_PATH="$LIBTBX_BUILD/lib:$LIBTBX_BUILD/base/lib"
  export DYLD_LIBRARY_PATH
fi

python /Users/Smeets/python/topasdiff/topasdiff.py "$@"