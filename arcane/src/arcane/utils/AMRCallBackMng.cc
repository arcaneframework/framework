// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRCallBackMng.cc                                              (C) 2000-2010 */
/*                                                                           */
/* Gestionnaire des callbacks.                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/AMRCallBackMng.h"



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// -------------------------------------------------------------------
void AMRCallBackMng::initialize()
{
}

// -------------------------------------------------------------------
void AMRCallBackMng::finalize()
{
	m_amr_transport_functors.clear();
}

// -------------------------------------------------------------------
void AMRCallBackMng::registerCallBack(IAMRTransportFunctor * f)
{
	m_amr_transport_functors.add(f);
}

// -------------------------------------------------------------------
void AMRCallBackMng::unregisterCallBack(IAMRTransportFunctor * f)
{
	m_amr_transport_functors.remove(f);
}
// -------------------------------------------------------------------
void AMRCallBackMng::callCallBacks(Array<ItemInternal*>& old, AMROperationType op)
{
	IAMRTransportFunctorList::const_iterator ib(m_amr_transport_functors.begin()), ie(m_amr_transport_functors.end());
	for (; ib != ie; ib++) {
		(*ib)->executeFunctor(old,op);
	}
}
// -------------------------------------------------------------------
void AMRCallBackMng::callCallBacks(Array<Cell>& old, AMROperationType op)
{
	IAMRTransportFunctorList::const_iterator ib(m_amr_transport_functors.begin()), ie(m_amr_transport_functors.end());
	for (; ib != ie; ib++) {
		(*ib)->executeFunctor(old,op);
	}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
