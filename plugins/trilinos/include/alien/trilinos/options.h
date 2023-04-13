/*
 * Copyright 2022 IFPEN-CEA
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

namespace Alien::Trilinos
{
struct OptionTypes
{
  enum eSolver
  {
    CG,
    GMRES,
    BICGSTAB
  };

  enum ePreconditioner
  {
    MueLu,
    Relaxation,
    NoPC
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

  Alien::Trilinos::OptionTypes::eSolver solver_ = Alien::Trilinos::OptionTypes::CG;
  Alien::Trilinos::OptionTypes::eSolver solver() const { return solver_; }
  Options& solver(Alien::Trilinos::OptionTypes::eSolver n)
  {
    solver_ = n;
    return *this;
  }

  Alien::Trilinos::OptionTypes::ePreconditioner preconditioner_ = Alien::Trilinos::OptionTypes::Relaxation;
  Alien::Trilinos::OptionTypes::ePreconditioner preconditioner() const { return preconditioner_; }
  Options& preconditioner(Alien::Trilinos::OptionTypes::ePreconditioner n)
  {
    preconditioner_ = n;
    return *this;
  }
};

class OptionsUtils
{
 public:
  static OptionTypes::eSolver stringToSolverEnum(const std::string& solver_s)
  {
    if (solver_s.compare("CG") == 0)
      return OptionTypes::CG;
    else if (solver_s.compare("GMRES") == 0)
      return OptionTypes::GMRES;
    else if (solver_s.compare("BICGSTAB") == 0)
      return OptionTypes::BICGSTAB;
    else
      throw Arccore::FatalErrorException(A_FUNCINFO, Arccore::String::format("solver enum name: {0} is not consistent with axl definition", solver_s));
  }

  static std::string solverEnumToString(const OptionTypes::eSolver& solver)
  {
    switch (solver) {
    case OptionTypes::CG:
      return "CG";
    case OptionTypes::GMRES:
      return "GMRES";
    case OptionTypes::BICGSTAB:
      return "BICGSTAB";
    default:
      throw Arccore::FatalErrorException(A_FUNCINFO, Arccore::String::format("Unmanaged TrilinosOptionTypes::eSolver value: {0}", solver));
    }
  }

  static OptionTypes::ePreconditioner stringToPreconditionerEnum(const std::string& preconditioner_s)
  {
    if (preconditioner_s.compare("none") == 0)
      return OptionTypes::NoPC;
    else if (preconditioner_s.compare("relaxation") == 0)
      return OptionTypes::Relaxation;
    else
      throw Arccore::FatalErrorException(A_FUNCINFO, Arccore::String::format("preconditioner enum name: {0} is not consistent with axl definition", preconditioner_s));
  }

  static std::string preconditionerEnumToString(const OptionTypes::ePreconditioner& preconditioner)
  {
    switch (preconditioner) {
    case OptionTypes::NoPC:
      return "none";
    case OptionTypes::Relaxation:
      return "relaxation";
    default:
      throw Arccore::FatalErrorException(A_FUNCINFO, Arccore::String::format("Unmanaged Trilinos OptionTypes::ePreconditioner value: {0}", preconditioner));
    }
  }
};

} // namespace Alien::Trilinos