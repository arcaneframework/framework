// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemEnumerator.cc                                           (C) 2000-2024 */
/*                                                                           */
/* Enumérateur sur des groupes d'entités du maillage.                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemEnumerator.h"

#include "arcane/utils/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
Ref<IItemEnumeratorTracer> m_singleton_tracer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemEnumeratorTracer* IItemEnumeratorTracer::
singleton()
{
  return m_singleton_tracer.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemEnumeratorPOD
{
 public:
  ItemSharedInfo* m_shared_info;
  const Int32* m_local_ids;
  Integer m_index;
  Integer m_count;
};

void
_arcaneInternalItemEnumeratorSwigSet(const ItemEnumerator* ie, ItemEnumeratorPOD* vpod)
{
  vpod-> m_local_ids = ie->unguardedLocalIds();
  vpod-> m_index = ie->index();
  vpod-> m_count = ie->count();
  vpod-> m_shared_info = ie->_sharedInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT void
arcaneSetSingletonItemEnumeratorTracer(Ref<IItemEnumeratorTracer> tracer)
{
  m_singleton_tracer = tracer;
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
