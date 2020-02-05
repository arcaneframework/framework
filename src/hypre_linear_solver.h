#pragma once

#include "hypre_backend.h"

#include <ALIEN/Core/Backend/IInternalLinearSolverT.h>
#include <ALIEN/Core/Backend/LinearSolver.h>

namespace Alien::Hypre {

  using LinearSolver = Alien::LinearSolver<BackEnd::tag::hypre>;

}
