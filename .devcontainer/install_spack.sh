#! /bin/bash

cat >> ~/.profile << EOF 
. /spack/share/spack/setup-env.sh && spack env activate alien
EOF
