// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentSimd.cc                                            (C) 2000-2017 */
/*                                                                           */
/* Support for vectorization for materials and environments.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/ComponentSimd.h"
#include "arcane/materials/EnvItemVector.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LoopFunctorEnvPartSimdCell LoopFunctorEnvPartSimdCell::
create(const EnvCellVector& env)
{
  return LoopFunctorEnvPartSimdCell(env.pureItems(), env.impureItems());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LoopFunctorEnvPartSimdCell LoopFunctorEnvPartSimdCell::
create(IMeshEnvironment* env)
{
  return LoopFunctorEnvPartSimdCell(env->pureItems(), env->impureItems());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
