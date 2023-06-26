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

namespace Alien::PETSc
{
struct OptionTypes
{
  enum eSolver
  {
    CG,
    GMRES,
    BiCG,
    BiCGstab
  };

  enum ePreconditioner
  {
    Jacobi,
    NoPC,
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

  Alien::PETSc::OptionTypes::eSolver solver_ = Alien::PETSc::OptionTypes::GMRES;

  Alien::PETSc::OptionTypes::eSolver solver() const { return solver_; }

  Options& solver(Alien::PETSc::OptionTypes::eSolver n)
  {
    solver_ = n;
    return *this;
  }

  Alien::PETSc::OptionTypes::ePreconditioner preconditioner_ = Alien::PETSc::OptionTypes::Jacobi;

  Alien::PETSc::OptionTypes::ePreconditioner preconditioner() const { return preconditioner_; }

  Options& preconditioner(Alien::PETSc::OptionTypes::ePreconditioner n)
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
    if (solver_s.compare("cg") == 0)
      return OptionTypes::CG;
    else if (solver_s.compare("gmres") == 0)
      return OptionTypes::GMRES;
    else if (solver_s.compare("bicg") == 0)
      return OptionTypes::BiCG;
    else if (solver_s.compare("bicgstab") == 0)
      return OptionTypes::BiCGstab;
    else
      throw Arccore::FatalErrorException(
      A_FUNCINFO,
      Arccore::String::format(
      "solver enum name: {0} is not consistent with axl definition",
      solver_s));
  }

  static std::string solverEnumToString(OptionTypes::eSolver solver)
  {
    switch (solver) {
    case OptionTypes::CG:
      return "cg";
    case OptionTypes::GMRES:
      return "gmres";
    case OptionTypes::BiCG:
      return "bicg";
    case OptionTypes::BiCGstab:
      return "bcgs";
    default:
      throw Arccore::FatalErrorException(A_FUNCINFO, Arccore::String::format("Unmanaged PetscOptionTypes::eSolver value: {0}", solver));
    }
  }

  static OptionTypes::ePreconditioner
  stringToPreconditionerEnum(const std::string& preconditioner_s)
  {
    if (preconditioner_s.compare("none") == 0)
      return OptionTypes::NoPC;
    else if (preconditioner_s.compare("jacobi") == 0)
      return OptionTypes::Jacobi;
    else
      throw Arccore::FatalErrorException(
      A_FUNCINFO,
      Arccore::String::format("preconditioner enum name: {0} is not "
                              "consistent with axl definition",
                              preconditioner_s));
  }

  static std::string preconditionerEnumToString(OptionTypes::ePreconditioner preconditioner)
  {
    switch (preconditioner) {
    case OptionTypes::NoPC:
      return "none";
    case OptionTypes::Jacobi:
      return "jacobi";
    default:
      throw Arccore::FatalErrorException(A_FUNCINFO, Arccore::String::format("Unmanaged PetscOptionTypes::ePreconditioner value: {0}", preconditioner));
    }
  }
};

} // namespace Alien::PETSc
