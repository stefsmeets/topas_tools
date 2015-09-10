#!/usr/bin/env bash

LIBTBX_BUILD="$LIBTBX_BUILD"

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

# http://stackoverflow.com/a/246128
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

python $DIR/../topas_tools/scripcif.py $@


