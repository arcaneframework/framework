﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Partitioner.cc                                              (C) 2000-2024 */
/*                                                                           */
/* Algorithme de partitionnement de liste.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/Partitioner.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GenericPartitionerBase::
GenericPartitionerBase()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 GenericPartitionerBase::
_nbFirstPart() const
{
  if (m_queue)
    m_queue->barrier();
  // Peut arriver si on n'a pas encore appelé _allocate()
  return m_host_nb_list1_storage[0];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GenericPartitionerBase::
_allocate()
{
  eMemoryRessource r = eMemoryRessource::Host;
  if (m_queue && isAcceleratorPolicy(m_queue->executionPolicy()))
    r = eMemoryRessource::HostPinned;
  if (m_host_nb_list1_storage.memoryRessource() != r)
    m_host_nb_list1_storage = NumArray<Int32, MDDim1>(r);
  m_host_nb_list1_storage.resize(1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
