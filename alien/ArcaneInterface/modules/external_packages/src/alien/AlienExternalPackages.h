// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <alien/AlienExternalPackagesPrecomp.h>

#ifdef ALIEN_USE_PETSC
#include <alien/functional/Dump.h>
#include <alien/kernels/petsc/algebra/PETScLinearAlgebra.h>
#include <alien/kernels/petsc/io/AsciiDumper.h>
#endif

#ifdef ALIEN_USE_MTL4
#include <alien/kernels/mtl/algebra/MTLLinearAlgebra.h>
#include <alien/kernels/mtl/linear_solver/arcane/MTLLinearSolverService.h>
#include <alien/kernels/mtl/linear_solver/MTLInternalLinearSolver.h>
#endif

#ifdef ALIEN_USE_PETSC
#include <alien/kernels/petsc/linear_solver/arcane/PETScLinearSolverService.h>
#include <alien/kernels/petsc/linear_solver/PETScInternalLinearSolver.h>
#endif

#ifdef ALIEN_USE_HYPRE
#include <alien/kernels/hypre/linear_solver/arcane/HypreLinearSolver.h>
#endif
