#pragma once

#include <alien/AlienExternalPackagesPrecomp.h>

#ifdef ALIEN_USE_PETSC
#include <alien/Functional/Dump.h>
#include <alien/Kernels/PETSc/Algebra/PETScLinearAlgebra.h>
#include <alien/Kernels/PETSc/IO/AsciiDumper.h>
#endif

#ifdef ALIEN_USE_MTL4
#include <alien/Kernels/MTL/Algebra/MTLLinearAlgebra.h>
#include <alien/Kernels/MTL/LinearSolver/Arcane/MTLLinearSolverService.h>
#include <alien/Kernels/MTL/LinearSolver/MTLInternalLinearSolver.h>
#endif

#ifdef ALIEN_USE_PETSC
#include <alien/Kernels/PETSc/LinearSolver/Arcane/PETScLinearSolverService.h>
#include <alien/Kernels/PETSc/LinearSolver/PETScInternalLinearSolver.h>
#endif

#ifdef ALIEN_USE_HYPRE
#include <alien/Kernels/Hypre/LinearSolver/Arcane/HypreLinearSolver.h>
#endif
