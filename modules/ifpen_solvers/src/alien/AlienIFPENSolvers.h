#pragma once

#include <alien/AlienIFPENSolversPrecomp.h>

#ifdef ALIEN_USE_IFPSOLVER
#include <alien/kernels/ifp/linear_solver/arcane/IFPLinearSolverService.h>
#include <alien/kernels/ifp/linear_solver/IFPInternalLinearSolver.h>
#endif

#ifdef ALIEN_USE_MCGSOLVER
#include <alien/kernels/mcg/linear_solver/arcane/MCGLinearSolver.h>
#endif
