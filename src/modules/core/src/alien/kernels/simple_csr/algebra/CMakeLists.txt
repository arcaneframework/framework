# Copyright 2020 IFPEN-CEA
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# SPDX-License-Identifier: Apache-2.0

find_package(BLAS REQUIRED)
target_link_libraries(alien_kernel_simplecsr INTERFACE ${BLAS_LIBRARIES})

SET(ALIEN_CBLAS_HEADER)

SET(ALIEN_BLAS_SDOT "sdot")
SET(ALIEN_BLAS_DDOT "ddot")
SET(ALIEN_BLAS_SAXPY "saxpy")
SET(ALIEN_BLAS_DAXPY "daxpy")
SET(ALIEN_BLAS_SCOPY "scopy")
SET(ALIEN_BLAS_DCOPY "dcopy")
SET(ALIEN_BLAS_SSCAL "sscal")
SET(ALIEN_BLAS_DSCAL "dscal")
SET(ALIEN_BLAS_DNRM2 "dnrm2")

IF (MKL_FOUND)
    SET(ALIEN_CBLAS_HEADER "mkl_cblas.h")
    SET(ALIEN_HAVE_CBLAS TRUE)
ELSE (MKL_FOUND)
    find_path(ALIEN_CBLAS_HPATH "cblas.h")
    if (NOT ALIEN_CBLAS_HPATH)
        SET(ALIEN_CBLAS_HEADER ${BLAS_CBLAS_HEADER})
    else ()
        SET(ALIEN_CBLAS_HEADER "cblas.h")
        SET(ALIEN_HAVE_CBLAS TRUE)
    endif ()
ENDIF (MKL_FOUND)

# We need to generate blas header file
IF (ALIEN_CBLAS_HEADER)
    SET(ALIEN_CBLAS_HEADER "<${ALIEN_CBLAS_HEADER}>")
ELSE (ALIEN_CBLAS_HEADER)
    IF (BLAS_HAVE_CBLAS)
        SET(ALIEN_CBLAS_NAME "cblas_decl.h")
        CONFIGURE_FILE(cblas_decl.h.in ${ALIEN_CBLAS_NAME} COPYONLY)
    ELSE (BLAS_HAVE_CBLAS)
        SET(ALIEN_CBLAS_NAME "alien_cblas.h")
        CONFIGURE_FILE(alien_cblas.h.in ${ALIEN_CBLAS_NAME} @ONLY)
    ENDIF (BLAS_HAVE_CBLAS)
    SET(ALIEN_CBLAS_HEADER "\"${ALIEN_CBLAS_NAME}\"")
ENDIF (ALIEN_CBLAS_HEADER)

CONFIGURE_FILE(CBLAS.h.in CBLAS.h @ONLY)
