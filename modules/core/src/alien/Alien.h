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

#warning "This file should not be used: include only what you use !"

#include <alien/utils/ICopyOnWriteObject.h>
#include <alien/utils/ObjectWithTrace.h>
#include <alien/utils/Precomp.h>
#include <alien/utils/parameter_manager/BaseParameterManager.h>
#include <alien/utils/time_stamp/TimestampMng.h>
#include <alien/utils/time_stamp/TimestampObserver.h>

#include <alien/data/ISpace.h>
#include <alien/data/Space.h>

#include <alien/data/CompositeMatrix.h>
#include <alien/data/IMatrix.h>

#include <alien/data/CompositeVector.h>
#include <alien/data/IVector.h>

#include <alien/data/Universal.h>
#include <alien/data/Universe.h>
#include <alien/data/UniverseDataBase.h>

#include <alien/data/utils/ExtractionIndices.h>

#include <alien/core/block/Block.h>
#include <alien/core/block/VBlock.h>
#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/core/utils/Partition.h>

#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>

#include <alien/handlers/block/BaseBlockVectorReader.h>
#include <alien/handlers/block/BaseBlockVectorWriter.h>
#include <alien/handlers/scalar/BaseVectorReader.h>
#include <alien/handlers/scalar/BaseVectorWriter.h>

#include <alien/handlers/profiler/BaseMatrixProfiler.h>
#include <alien/handlers/scalar/BaseDirectMatrixBuilder.h>
#include <alien/handlers/scalar/BaseProfiledMatrixBuilder.h>

#include <alien/functional/Cast.h>

#include <alien/expression/normalization/NormalizeOpt.h>
#include <alien/expression/solver/ILinearSolver.h>
#include <alien/expression/solver/SolverStat.h>

#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>

#include <alien/kernels/composite/CompositeMultiVectorImpl.h>
#include <alien/kernels/composite/CompositeSpace.h>

#include <alien/kernels/dok/DoKBackEnd.h>
#include <alien/kernels/dok/DoKMatrixT.h>
