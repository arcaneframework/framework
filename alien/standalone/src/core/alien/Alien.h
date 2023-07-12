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

#include "utils/ICopyOnWriteObject.h"
#include "utils/ObjectWithTrace.h"
#include "utils/Precomp.h"
#include "utils/parameter_manager/BaseParameterManager.h"
#include "utils/time_stamp/TimestampMng.h"
#include "utils/time_stamp/TimestampObserver.h"

#include "data/ISpace.h"
#include "data/Space.h"

#include "data/CompositeMatrix.h"
#include "data/IMatrix.h"

#include "data/CompositeVector.h"
#include "data/IVector.h"

#include "data/Universal.h"
#include "data/Universe.h"
#include "data/UniverseDataBase.h"

#include "data/utils/ExtractionIndices.h"

#include "core/block/Block.h"
#include "core/block/VBlock.h"
#include "core/impl/MultiMatrixImpl.h"
#include "core/impl/MultiVectorImpl.h"
#include "core/utils/Partition.h"

#include "distribution/MatrixDistribution.h"
#include "distribution/VectorDistribution.h"

#include "handlers/block/BaseBlockVectorReader.h"
#include "handlers/block/BaseBlockVectorWriter.h"
#include "handlers/scalar/BaseVectorReader.h"
#include "handlers/scalar/BaseVectorWriter.h"

#include "handlers/profiler/BaseMatrixProfiler.h"
#include "handlers/scalar/BaseDirectMatrixBuilder.h"
#include "handlers/scalar/BaseProfiledMatrixBuilder.h"

#include "functional/Cast.h"

#include "expression/normalization/NormalizeOpt.h"
#include "expression/solver/ILinearSolver.h"
#include "expression/solver/SolverStat.h"

#include "kernels/simple_csr/SimpleCSRMatrix.h"
#include "kernels/simple_csr/SimpleCSRVector.h"
#include "kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h"

#include "kernels/composite/CompositeMultiVectorImpl.h"
#include "kernels/composite/CompositeSpace.h"

#include "kernels/dok/DoKBackEnd.h"
#include "kernels/dok/DoKMatrixT.h"
