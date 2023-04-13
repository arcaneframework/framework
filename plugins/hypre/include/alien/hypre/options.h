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

#include <string>
#include <arccore/base/ArccoreGlobal.h>
#include <arccore/base/FatalErrorException.h>

namespace Alien::Hypre
{
struct OptionTypes
{
  enum eSolver
  {
    AMG,
    CG,
    GMRES,
    BiCGStab,
    Hybrid
  };

  enum ePreconditioner
  {
    NoPC,
    DiagPC,
    AMGPC,
    ParaSailsPC,
    EuclidPC
  };

  enum class eProblem
  {
    Geometric_2D,
    Geometric_3D,
    Default
  };
};

struct Options
{
  Arccore::Integer numIterationsMax_ = 100;
  Arccore::Integer numIterationsMax() const { return numIterationsMax_; }
  Options& numIterationsMax(Arccore::Integer n)
  {
    numIterationsMax_ = n;
    return *this;
  }

  Arccore::Real stopCriteriaValue_ = 1.e-10;
  Arccore::Real stopCriteriaValue() const { return stopCriteriaValue_; }
  Options& stopCriteriaValue(Arccore::Real n)
  {
    stopCriteriaValue_ = n;
    return *this;
  }

  bool verbose_ = false;
  bool verbose() const { return verbose_; }
  Options& verbose(bool n)
  {
    verbose_ = n;
    return *this;
  }

  Alien::Hypre::OptionTypes::eSolver solver_ = Alien::Hypre::OptionTypes::eSolver::GMRES;
  Alien::Hypre::OptionTypes::eSolver solver() const { return solver_; }
  Options& solver(Alien::Hypre::OptionTypes::eSolver n)
  {
    solver_ = n;
    return *this;
  }

  Alien::Hypre::OptionTypes::ePreconditioner preconditioner_ = Alien::Hypre::OptionTypes::ePreconditioner::AMGPC;
  Alien::Hypre::OptionTypes::ePreconditioner preconditioner() const { return preconditioner_; }
  Options& preconditioner(Alien::Hypre::OptionTypes::ePreconditioner n)
  {
    preconditioner_ = n;
    return *this;
  }

  Alien::Hypre::OptionTypes::eProblem problem_ = Alien::Hypre::OptionTypes::eProblem::Default;
  auto problemKind() const { return problem_; }
  Options& problemKind(Alien::Hypre::OptionTypes::eProblem n)
  {
    problem_ = n;
    return *this;
  }
};

class OptionsUtils
{
 public:
  static OptionTypes::eSolver stringToSolverEnum(const std::string& solver_s)
  {
    if (solver_s.compare("amg") == 0)
      return OptionTypes::eSolver::AMG;
    else if (solver_s.compare("cg") == 0)
      return OptionTypes::eSolver::CG;
    else if (solver_s.compare("gmres") == 0)
      return OptionTypes::eSolver::GMRES;
    else if (solver_s.compare("bicgstab") == 0)
      return OptionTypes::eSolver::BiCGStab;
    else if (solver_s.compare("hybrid") == 0)
      return OptionTypes::eSolver::Hybrid;
    else
      throw Arccore::FatalErrorException(A_FUNCINFO, Arccore::String::format("solver enum name: {0} is not consistent with axl definition", solver_s));
  }

  static std::string solverEnumToString(OptionTypes::eSolver solver)
  {
    switch (solver) {
    case OptionTypes::eSolver::AMG:
      return "amg";
    case OptionTypes::eSolver::CG:
      return "cg";
    case OptionTypes::eSolver::GMRES:
      return "gmres";
    case OptionTypes::eSolver::BiCGStab:
      return "bicgstab";
    case OptionTypes::eSolver::Hybrid:
      return "hybrid";
    default:
      throw Arccore::FatalErrorException(A_FUNCINFO, Arccore::String::format("Unmanaged HypreOptionTypes::eSolver"));
    }
  }

  static OptionTypes::ePreconditioner stringToPreconditionerEnum(const std::string& preconditioner_s)
  {
    if (preconditioner_s.compare("none") == 0)
      return OptionTypes::ePreconditioner::NoPC;
    else if (preconditioner_s.compare("diag") == 0)
      return OptionTypes::ePreconditioner::DiagPC;
    else if (preconditioner_s.compare("amg") == 0)
      return OptionTypes::ePreconditioner::AMGPC;
    else if (preconditioner_s.compare("parasails") == 0)
      return OptionTypes::ePreconditioner::ParaSailsPC;
    else if (preconditioner_s.compare("euclid") == 0)
      return OptionTypes::ePreconditioner::EuclidPC;
    else
      throw Arccore::FatalErrorException(A_FUNCINFO, Arccore::String::format("preconditioner enum name: {0} is not consistent with axl definition", preconditioner_s));
  }

  static std::string preconditionerEnumToString(OptionTypes::ePreconditioner preconditioner)
  {
    switch (preconditioner) {
    case OptionTypes::ePreconditioner::NoPC:
      return "none";
    case OptionTypes::ePreconditioner::DiagPC:
      return "diag";
    case OptionTypes::ePreconditioner::AMGPC:
      return "amg";
    case OptionTypes::ePreconditioner::ParaSailsPC:
      return "parasails";
    case OptionTypes::ePreconditioner::EuclidPC:
      return "euclid";
    default:
      throw Arccore::FatalErrorException(A_FUNCINFO, Arccore::String::format("Unmanaged HypreOptionTypes::ePreconditioner"));
    }
  }
};

} // namespace Alien::Hypre