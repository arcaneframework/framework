/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// We can discuss that point
// #warning "This file should not be used ?"

#include <alien/ref/data/block/BlockMatrix.h>
#include <alien/ref/data/block/BlockVector.h>
#include <alien/ref/data/block/VBlockMatrix.h>
#include <alien/ref/data/block/VBlockVector.h>
#include <alien/ref/data/scalar/Matrix.h>
#include <alien/ref/data/scalar/Vector.h>

#include <alien/ref/handlers/block/BlockVectorReader.h>
#include <alien/ref/handlers/block/BlockVectorWriter.h>
#include <alien/ref/handlers/block/VBlockVectorReader.h>
#include <alien/ref/handlers/block/VBlockVectorWriter.h>
#include <alien/ref/handlers/scalar/VectorReader.h>
#include <alien/ref/handlers/scalar/VectorWriter.h>

#include <alien/ref/handlers/block/ProfiledBlockMatrixBuilder.h>
#include <alien/ref/handlers/block/ProfiledVBlockMatrixBuilder.h>
#include <alien/ref/handlers/profiler/MatrixProfiler.h>
#include <alien/ref/handlers/scalar/DirectMatrixBuilder.h>
#include <alien/ref/handlers/scalar/ProfiledMatrixBuilder.h>
#include <alien/ref/handlers/stream/StreamMatrixBuilder.h>
#include <alien/ref/handlers/stream/StreamMatrixBuilderInserter.h>
#include <alien/ref/handlers/stream/StreamVBlockMatrixBuilder.h>
#include <alien/ref/handlers/stream/StreamVBlockMatrixBuilderInserter.h>

#include <alien/ref/functional/Ones.h>
#include <alien/ref/functional/Zeros.h>

#include <alien/ref/distribution/DistributionFabric.h>
