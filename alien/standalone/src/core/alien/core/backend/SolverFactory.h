// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SolverFactory                                  (C) 2000-2024              */
/*                                                                           */
/* Factory for Alien solvers                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#pragma once

#include <alien/core/backend/ISolverFabric.h>
#include <alien/core/backend/SolverFabricRegisterer.h>
#include <alien/utils/Precomp.h>

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
