// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupImplInternal.h                                     (C) 2000-2024 */
/*                                                                           */
/* API interne à Arcane de ItemGroupImpl.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_ITEMGROUPIMPLINTERNAL_H
#define ARCANE_CORE_INTERNAL_ITEMGROUPIMPLINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemGroupImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief API interne à Arcane de ItemGroupImpl
 */
class ItemGroupImplInternal
{
 public:

  ItemGroupImplInternal(ItemGroupInternal* p)
  : m_p(p)
  {}

 private:

  ItemGroupInternal* m_p = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
