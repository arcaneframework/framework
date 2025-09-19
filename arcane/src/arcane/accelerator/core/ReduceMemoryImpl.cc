// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ReduceMemoryImpl.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Gestion de la mémoire pour les réductions.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/internal/ReduceMemoryImpl.h"

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/core/internal/IRunQueueStream.h"
#include "arcane/accelerator/core/internal/RunCommandImpl.h"
#include "arcane/accelerator/core/internal/RunnerImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{
namespace
{
  IMemoryAllocator* _getAllocator(eMemoryRessource r)
  {
    return MemoryUtils::getAllocator(r);
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ReduceMemoryImpl::
ReduceMemoryImpl(RunCommandImpl* p)
: m_command(p)
, m_device_memory_bytes(_getAllocator(eMemoryRessource::Device))
, m_host_memory_bytes(_getAllocator(eMemoryRessource::HostPinned))
, m_grid_buffer(_getAllocator(eMemoryRessource::Device))
, m_grid_device_count(_getAllocator(eMemoryRessource::Device))
{
  _allocateMemoryForReduceData(128);
  _allocateMemoryForGridDeviceCount();
  m_grid_memory_info.m_warp_size = p->runner()->deviceInfo().warpSize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ReduceMemoryImpl::
release()
{
  m_command->releaseReduceMemoryImpl(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* ReduceMemoryImpl::
allocateReduceDataMemory(ConstMemoryView identity_view)
{
  auto identity_span = identity_view.bytes();
  Int32 data_type_size = static_cast<Int32>(identity_span.size());
  m_data_type_size = data_type_size;
  if (data_type_size > m_size)
    _allocateMemoryForReduceData(data_type_size);

  // Recopie \a identity_view dans un buffer car on utilise l'asynchronisme
  // et la zone pointée par \a identity_view n'est pas forcément conservée
  m_identity_buffer.copy(identity_view.bytes());
  MemoryCopyArgs copy_args(m_device_memory, m_identity_buffer.span().data(), data_type_size);
  m_command->internalStream()->copyMemory(copy_args.addAsync());

  return m_device_memory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ReduceMemoryImpl::
_allocateGridDataMemory()
{
  // TODO: pouvoir utiliser un padding pour éviter que les lignes de cache
  // entre les blocs se chevauchent
  Int32 total_size = CheckedConvert::toInt32(m_data_type_size * m_grid_size);
  if (total_size <= m_grid_memory_info.m_grid_memory_values.bytes().size())
    return;

  m_grid_buffer.resize(total_size);

  auto mem_view = makeMutableMemoryView(m_grid_buffer.span());
  m_grid_memory_info.m_grid_memory_values = mem_view;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ReduceMemoryImpl::
_allocateMemoryForGridDeviceCount()
{
  // Alloue sur le device la mémoire contenant le nombre de blocs restant à traiter
  // Il s'agit d'un seul entier non signé.
  Int64 size = sizeof(unsigned int);
  const unsigned int zero = 0;
  m_grid_device_count.resize(1);
  auto* ptr = m_grid_device_count.data();

  m_grid_memory_info.m_grid_device_count = ptr;

  // Initialise cette zone mémoire avec 0.
  MemoryCopyArgs copy_args(ptr, &zero, size);
  m_command->internalStream()->copyMemory(copy_args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ReduceMemoryImpl::
copyReduceValueFromDevice()
{
  void* destination = m_grid_memory_info.m_host_memory_for_reduced_value;
  void* source = m_device_memory;
  MemoryCopyArgs copy_args(destination, source, m_data_type_size);
  m_command->internalStream()->copyMemory(copy_args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IReduceMemoryImpl*
internalGetOrCreateReduceMemoryImpl(RunCommand* command)
{
  return command->m_p->getOrCreateReduceMemoryImpl();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
