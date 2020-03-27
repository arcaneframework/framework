#pragma once

#include <ALIEN/AlienExternalPackagesPrecomp.h>

#ifdef ALIEN_USE_PETSC
#include <ALIEN/Functional/Dump.h>
#include <ALIEN/Kernels/PETSc/Algebra/PETScLinearAlgebra.h>
#include <ALIEN/Kernels/PETSc/IO/AsciiDumper.h>
#endif

#ifdef ALIEN_USE_MTL4
#include <ALIEN/Kernels/MTL/Algebra/MTLLinearAlgebra.h>
#include <ALIEN/Kernels/MTL/LinearSolver/Arcane/MTLLinearSolverService.h>
#include <ALIEN/Kernels/MTL/LinearSolver/MTLInternalLinearSolver.h>
#endif

#ifdef ALIEN_USE_PETSC
#include <ALIEN/Kernels/PETSc/LinearSolver/Arcane/PETScLinearSolverService.h>
#include <ALIEN/Kernels/PETSc/LinearSolver/PETScInternalLinearSolver.h>
#endif

#ifdef ALIEN_USE_HYPRE
#include <ALIEN/Kernels/Hypre/LinearSolver/Arcane/HypreLinearSolver.h>
#endif
