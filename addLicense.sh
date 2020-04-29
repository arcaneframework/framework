#!/bin/bash

DIRNAME=$(dirname $(readlink -f $0))
COPYRIGHT_FILE=${DIRNAME}/copyright

DIR=$(pwd)
if [[ -d "$1" ]] ; then
  DIR=$1;
fi

FILE_LIST=$(mktemp)
grep -r -L Copyright ${DIR} >${FILE_LIST}

REWRITTEN_FILE=$(mktemp)

for f in $(grep -E '^.*(\.cc|\.cpp|\.h)$' ${FILE_LIST}); do
  echo -e " */\n" | cat - $f >${REWRITTEN_FILE}
  {
    echo "/*"
    sed 's/.*/ * &/' ${COPYRIGHT_FILE}
  } | cat - ${REWRITTEN_FILE} >$f
done

for f in $(grep CMakeLists.txt ${FILE_LIST}); do
  echo -n -e "\n" | cat - $f >${REWRITTEN_FILE}
  sed 's/.*/# &/' ${COPYRIGHT_FILE} | cat - ${REWRITTEN_FILE} >$f
done

if [[ -f ${FILE_LIST} ]]; then
  rm -f ${FILE_LIST}
fi

if [[ -f ${REWRITTEN_FILE} ]]; then
  rm -f ${REWRITTEN_FILE}
fi
