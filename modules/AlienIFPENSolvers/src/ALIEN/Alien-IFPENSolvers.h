#pragma once

#include <ALIEN/Alien-IFPENSolversPrecomp.h>

#ifdef ALIEN_USE_IFPSOLVER
#include <ALIEN/Kernels/IFP/LinearSolver/Arcane/IFPLinearSolverService.h>
#include <ALIEN/Kernels/IFP/LinearSolver/IFPInternalLinearSolver.h>
#endif

#ifdef ALIEN_USE_MCGSOLVER
#include <ALIEN/Kernels/MCG/LinearSolver/Arcane/GPULinearSolver.h>
#endif
