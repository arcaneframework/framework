#pragma once

#include <alien/AlienIFPENSolversPrecomp.h>

#ifdef ALIEN_USE_IFPSOLVER
#include <alien/Kernels/IFP/LinearSolver/Arcane/IFPLinearSolverService.h>
#include <alien/Kernels/IFP/LinearSolver/IFPInternalLinearSolver.h>
#endif

#ifdef ALIEN_USE_MCGSOLVER
#include <alien/Kernels/MCG/LinearSolver/Arcane/GPULinearSolver.h>
#endif
