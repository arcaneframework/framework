// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataSynchronizeBuffer.cc                                    (C) 2000-2023 */
/*                                                                           */
/* Implémentation d'un buffer générique pour la synchronisation de donnéess. */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/DataSynchronizeBuffer.h"

#include "arcane/impl/IBufferCopier.h"
#include "arcane/impl/DataSynchronizeInfo.h"

#include "arcane/accelerator/core/Runner.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IDataSynchronizeBuffer::
copyAllSend()
{
  Int32 nb_rank = nbRank();
  for (Int32 i = 0; i < nb_rank; ++i)
    copySendAsync(i);
  barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IDataSynchronizeBuffer::
copyAllReceive()
{
  Int32 nb_rank = nbRank();
  for (Int32 i = 0; i < nb_rank; ++i)
    copyReceiveAsync(i);
  barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DirectBufferCopier::
barrier()
{
  if (m_queue)
    m_queue->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizeBufferBase::
barrier()
{
  m_buffer_copier->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 VariableSynchronizeBufferBase::
_ghostDisplacementBase(Int32 index) const
{
  return m_sync_info->ghostDisplacement(index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 VariableSynchronizeBufferBase::
_shareDisplacementBase(Int32 index) const
{
  return m_sync_info->shareDisplacement(index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 VariableSynchronizeBufferBase::
_nbGhost(Int32 index) const
{
  return m_sync_info->rankInfo(index).nbGhost();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 VariableSynchronizeBufferBase::
_nbShare(Int32 index) const
{
  return m_sync_info->rankInfo(index).nbShare();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MutableMemoryView VariableSynchronizeBufferBase::
_shareLocalBuffer(Int32 index) const
{
  Int64 displacement = _shareDisplacementBase(index);
  Int32 local_size = _nbShare(index);
  return m_share_memory_view.subView(displacement, local_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MutableMemoryView VariableSynchronizeBufferBase::
_ghostLocalBuffer(Int32 index) const
{
  Int64 displacement = _ghostDisplacementBase(index);
  Int32 local_size = _nbGhost(index);
  return m_ghost_memory_view.subView(displacement, local_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul et alloue les tampons nécessaire aux envois et réceptions
 * pour les synchronisations des variables 1D.
 */
void VariableSynchronizeBufferBase::
compute(IBufferCopier* copier, DataSynchronizeInfo* sync_info, Int32 datatype_size)
{
  m_datatype_size = datatype_size;
  m_buffer_copier = copier;
  m_sync_info = sync_info;
  m_nb_rank = sync_info->size();

  IMemoryAllocator* allocator = m_buffer_copier->allocator();
  if (allocator && allocator != m_buffer.allocator())
    m_buffer = UniqueArray<std::byte>(allocator);

  _allocateBuffers(datatype_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul et alloue les tampons nécessaires aux envois et réceptions
 * pour les synchronisations des variables 1D.
 *
 * \todo: ne pas converver les tampons pour chaque type de donnée des variables
 * car leur conservation est couteuse en terme de memoire.
 */
void VariableSynchronizeBufferBase::
_allocateBuffers(Int32 datatype_size)
{
  Int64 total_ghost_buffer = m_sync_info->totalNbGhost();
  Int64 total_share_buffer = m_sync_info->totalNbShare();

  Int32 full_dim2_size = datatype_size;
  m_buffer.resize((total_ghost_buffer + total_share_buffer) * full_dim2_size);

  Int64 share_offset = total_ghost_buffer * full_dim2_size;

  auto s1 = m_buffer.span().subspan(0, share_offset);
  m_ghost_memory_view = makeMutableMemoryView(s1.data(), full_dim2_size, total_ghost_buffer);
  auto s2 = m_buffer.span().subspan(share_offset, total_share_buffer * full_dim2_size);
  m_share_memory_view = makeMutableMemoryView(s2.data(), full_dim2_size, total_share_buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SingleDataSynchronizeBuffer::
copyReceiveAsync(Int32 index)
{
  ARCANE_CHECK_POINTER(m_sync_info);
  ARCANE_CHECK_POINTER(m_buffer_copier);

  MutableMemoryView var_values = dataView();
  ConstArrayView<Int32> indexes = m_sync_info->rankInfo(index).ghostIds();
  ConstMemoryView local_buffer = receiveBuffer(index);

  m_buffer_copier->copyFromBufferAsync(indexes, local_buffer, var_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SingleDataSynchronizeBuffer::
copySendAsync(Int32 index)
{
  ARCANE_CHECK_POINTER(m_sync_info);
  ARCANE_CHECK_POINTER(m_buffer_copier);

  ConstMemoryView var_values = dataView();
  Int32ConstArrayView indexes = m_sync_info->rankInfo(index).shareIds();
  MutableMemoryView local_buffer = sendBuffer(index);
  m_buffer_copier->copyToBufferAsync(indexes, local_buffer, var_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiDataSynchronizeBuffer::
copyReceiveAsync(Int32 index)
{
  ARCANE_CHECK_POINTER(m_sync_info);
  ARCANE_CHECK_POINTER(m_buffer_copier);

  Int64 data_offset = 0;
  Span<const std::byte> local_buffer_bytes = receiveBuffer(index).bytes();
  Int32ConstArrayView indexes = m_sync_info->rankInfo(index).ghostIds();
  const Int64 nb_element = indexes.size();
  for (MutableMemoryView var_values : m_data_views) {
    Int32 datatype_size = var_values.datatypeSize();
    Int64 current_size_in_bytes = nb_element * datatype_size;
    Span<const std::byte> sub_local_buffer_bytes = local_buffer_bytes.subSpan(data_offset, current_size_in_bytes);
    ConstMemoryView local_buffer = makeConstMemoryView(sub_local_buffer_bytes.data(), datatype_size, nb_element);
    if (current_size_in_bytes != 0)
      m_buffer_copier->copyFromBufferAsync(indexes, local_buffer, var_values);
    data_offset += current_size_in_bytes;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiDataSynchronizeBuffer::
copySendAsync(Int32 index)
{
  ARCANE_CHECK_POINTER(m_sync_info);
  ARCANE_CHECK_POINTER(m_buffer_copier);

  Int64 data_offset = 0;
  Span<std::byte> local_buffer_bytes = sendBuffer(index).bytes();
  Int32ConstArrayView indexes = m_sync_info->rankInfo(index).shareIds();
  const Int64 nb_element = indexes.size();
  for (ConstMemoryView var_values : m_data_views) {
    Int32 datatype_size = var_values.datatypeSize();
    Int64 current_size_in_bytes = nb_element * datatype_size;
    Span<std::byte> sub_local_buffer_bytes = local_buffer_bytes.subSpan(data_offset, current_size_in_bytes);
    MutableMemoryView local_buffer = makeMutableMemoryView(sub_local_buffer_bytes.data(), datatype_size, nb_element);
    if (current_size_in_bytes != 0)
      m_buffer_copier->copyToBufferAsync(indexes, local_buffer, var_values);
    data_offset += current_size_in_bytes;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
