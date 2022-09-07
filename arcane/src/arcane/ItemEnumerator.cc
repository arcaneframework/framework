// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemEnumerator.cc                                           (C) 2000-2022 */
/*                                                                           */
/* Enumérateur sur des groupes d'entités du maillage.                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemEnumerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemEnumeratorTracer* IItemEnumeratorTracer::m_singleton = nullptr;

void IItemEnumeratorTracer::
_setSingleton(IItemEnumeratorTracer* tracer)
{
  delete m_singleton;
  m_singleton = tracer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Pour tester la validité des instantiations

template class ItemVectorViewConstIteratorT<Cell>;
template class ItemVectorViewT<Cell>;

template class ItemEnumeratorT<Node>;
template class ItemEnumeratorT<Edge>;
template class ItemEnumeratorT<Face>;
template class ItemEnumeratorT<Cell>;
template class ItemEnumeratorT<Particle>;
template class ItemEnumeratorT<DoF>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
