// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Filtering.cc                                                (C) 2000-2023 */
/*                                                                           */
/* Algorithme de filtrage.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/Filter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GenericFilteringBase::
GenericFilteringBase()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 GenericFilteringBase::
_nbOutputElement() const
{
  if (m_queue)
    m_queue->barrier();
  // Peut arriver si on n'a pas encore appelé _allocate()
  //if (m_host_nb_out_storage.totalNbElement()==0)
  //ARCANE_FATAL("Can not get output return 0;
  return m_host_nb_out_storage[0];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GenericFilteringBase::
_allocate()
{
  eMemoryRessource r = eMemoryRessource::Host;
  if (m_queue && isAcceleratorPolicy(m_queue->executionPolicy()))
    r = eMemoryRessource::HostPinned;
  if (m_host_nb_out_storage.memoryRessource()!=r)
    m_host_nb_out_storage = NumArray<Int32,MDDim1>(r);
  m_host_nb_out_storage.resize(1);    
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
