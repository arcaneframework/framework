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

namespace Alien::Ginkgo
{
struct OptionTypes
{
  enum eSolver
  {
    CG,
    GMRES,
    BICG,
    BICGSTAB
  };

  enum ePreconditioner
  {
    Jacobi,
    Ilu,
    NoPC,
  };
};

struct Options
{
  // attributes
  Arccore::Integer numIterationsMax_ = 100;
  Arccore::Real stopCriteriaValue_ = 1.e-10;
  bool verbose_ = false;
  Alien::Ginkgo::OptionTypes::eSolver solver_ = Alien::Ginkgo::OptionTypes::CG;
  Alien::Ginkgo::OptionTypes::ePreconditioner preconditioner_ = Alien::Ginkgo::OptionTypes::Jacobi;
  Arccore::Integer blockSize_ = 1;

  // methods
  Arccore::Integer numIterationsMax() const { return numIterationsMax_; }
  Options& numIterationsMax(Arccore::Integer n)
  {
    numIterationsMax_ = n;
    return *this;
  }

  Arccore::Real stopCriteriaValue() const { return stopCriteriaValue_; }
  Options& stopCriteriaValue(Arccore::Real n)
  {
    stopCriteriaValue_ = n;
    return *this;
  }

  bool verbose() const { return verbose_; }
  Options& verbose(bool n)
  {
    verbose_ = n;
    return *this;
  }

  Alien::Ginkgo::OptionTypes::eSolver solver() const { return solver_; }
  Options& solver(Alien::Ginkgo::OptionTypes::eSolver n)
  {
    solver_ = n;
    return *this;
  }

  Alien::Ginkgo::OptionTypes::ePreconditioner preconditioner() const { return preconditioner_; }
  Options& preconditioner(Alien::Ginkgo::OptionTypes::ePreconditioner n)
  {
    preconditioner_ = n;
    return *this;
  }

  Arccore::Integer blockSize() const { return blockSize_; }
  Options& blockSize(Arccore::Integer n)
  {
    blockSize_ = n;
    return *this;
  }
};
} // namespace Alien::Ginkgo
