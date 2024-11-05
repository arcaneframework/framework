﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Filtering.cc                                                (C) 2000-2024 */
/*                                                                           */
/* Algorithme de filtrage.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/Filter.h"

#include "arcane/utils/ValueConvert.h"

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
  m_queue.barrier();
  return m_host_nb_out_storage[0];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GenericFilteringBase::
_allocate()
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_FILTERER_USE_HOSTPINNED_STORAGE", true))
    m_use_direct_host_storage = (v.value() != 0);

  // Pour l'instant l'usage direct de l'hôte n'est testé qu'avec CUDA.
  if (m_queue.executionPolicy() != eExecutionPolicy::CUDA)
    m_use_direct_host_storage = false;

  eMemoryRessource r = eMemoryRessource::HostPinned;
  if (m_host_nb_out_storage.memoryRessource() != r)
    m_host_nb_out_storage = NumArray<Int32, MDDim1>(r);
  m_host_nb_out_storage.resize(1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GenericFilteringBase::
_allocateTemporaryStorage(size_t size)
{
  m_algo_storage.allocate(size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int* GenericFilteringBase::
_getDeviceNbOutPointer()
{
  if (m_use_direct_host_storage)
    return m_host_nb_out_storage.to1DSpan().data();

  return m_device_nb_out_storage.allocate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GenericFilteringBase::
_copyDeviceNbOutToHostNbOut()
{
  if (!m_use_direct_host_storage)
    m_device_nb_out_storage.copyToAsync(m_host_nb_out_storage, m_queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
