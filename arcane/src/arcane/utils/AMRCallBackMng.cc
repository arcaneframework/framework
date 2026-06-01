// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRCallBackMng.cc                                           (C) 2000-2010 */
/*                                                                           */
/* Callback Manager.                                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/AMRCallBackMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCallBackMng::
initialize()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCallBackMng::
finalize()
{
  m_amr_transport_functors.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCallBackMng::
registerCallBack(IAMRTransportFunctor* f)
{
  m_amr_transport_functors.add(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCallBackMng::
unregisterCallBack(IAMRTransportFunctor* f)
{
  m_amr_transport_functors.remove(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCallBackMng::
callCallBacks(Array<ItemInternal*>& old, AMROperationType op)
{
  IAMRTransportFunctorList::const_iterator ib(m_amr_transport_functors.begin()), ie(m_amr_transport_functors.end());
  for (; ib != ie; ib++) {
    (*ib)->executeFunctor(old, op);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCallBackMng::
callCallBacks(Array<Cell>& old, AMROperationType op)
{
  IAMRTransportFunctorList::const_iterator ib(m_amr_transport_functors.begin()), ie(m_amr_transport_functors.end());
  for (; ib != ie; ib++) {
    (*ib)->executeFunctor(old, op);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
