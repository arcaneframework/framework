// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Partitioner.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Algorithme de partitionnement de liste.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/GenericPartitioner.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GenericPartitionerBase::
GenericPartitionerBase(const RunQueue& queue)
: m_queue(queue)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 GenericPartitionerBase::
_nbFirstPart() const
{
  m_queue.barrier();
  return m_host_nb_list1_storage[0];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SmallSpan<const Int32> GenericPartitionerBase::
_nbParts() const
{
  m_queue.barrier();
  return m_host_nb_list1_storage.to1DSmallSpan();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GenericPartitionerBase::
_allocate()
{
  eMemoryResource r = eMemoryResource::HostPinned;
  if (m_host_nb_list1_storage.memoryRessource() != r)
    m_host_nb_list1_storage = NumArray<Int32, MDDim1>(r);
  // Il faut deux valeurs pour la version qui décompose la liste en trois
  // parties
  m_host_nb_list1_storage.resize(2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GenericPartitionerBase::
_resetNbPart()
{
  m_host_nb_list1_storage.fill(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
