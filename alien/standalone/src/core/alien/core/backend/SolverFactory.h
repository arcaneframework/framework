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

/*!
 * \file SolverFactory.h
 * \brief SolverFactory.h
 */
#pragma once

#include <alien/core/backend/ISolverFabric.h>
#include <alien/core/backend/SolverFabricRegisterer.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_EXPORT SolverFactory
{
 public:
  static void add_options(
  BackEndId back_end, ISolverFabric::CmdLineOptionDescType& cmdline_options)
  {
    auto* fabric = SolverFabricRegisterer::getSolverFabric(back_end);
    if (fabric)
      fabric->add_options(cmdline_options);
  }

  static ILinearSolver* create(BackEndId back_end,
                               ISolverFabric::CmdLineOptionType const& options, IMessagePassingMng* pm)
  {
    auto* fabric = SolverFabricRegisterer::getSolverFabric(back_end);
    if (fabric)
      return fabric->create(options, pm);
    else
      return nullptr;
  }

  static ILinearSolver* create(BackEndId back_end,
                               ISolverFabric::JsonOptionType const& options, IMessagePassingMng* pm)
  {
    auto* fabric = SolverFabricRegisterer::getSolverFabric(back_end);
    if (fabric)
      return fabric->create(options, pm);
    else
      return nullptr;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
