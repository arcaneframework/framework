// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataSynchronizeBuffer.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Implémentation d'un buffer générique pour la synchronisation de données.  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/DataSynchronizeBuffer.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/internal/MemoryBuffer.h"

#include "arcane/impl/DataSynchronizeInfo.h"
#include "arcane/impl/internal/IBufferCopier.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/utils/FixedArray.h"
#include "arccore/trace/ITraceMng.h"

#include <cstddef>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  //! Alignement pour les buffers et les sous-parties des buffers
  constexpr Int64 ALIGNEMENT_SIZE = 64;
  Int64 _applyPadding(Int64 original_size)
  {
    Int64 modulo = original_size % ALIGNEMENT_SIZE;
    Int64 new_size = original_size;
    if (modulo != 0)
      new_size += (ALIGNEMENT_SIZE - modulo);
    if ((new_size % ALIGNEMENT_SIZE) != 0)
      ARCANE_FATAL("Bad padding");
    return new_size;
  }
  void _checkAlignment(const void* address)
  {
    auto a = reinterpret_cast<intptr_t>(address);
    intptr_t max_align = alignof(std::max_align_t);
    intptr_t modulo = a % max_align;
    if (modulo != 0)
      ARCANE_FATAL("Address '{0}' is not aligned (align={1}, modulo={2})", address, max_align, modulo);
  }
} // namespace

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
  m_queue.barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Positionne le buffer global.
void DataSynchronizeBufferBase::BufferInfo::
setGlobalBuffer(MutableMemoryView v)
{
  if (v.datatypeSize() != 1)
    ARCANE_FATAL("Global buffer has to use a datatype of size 1 (current={0})", v.datatypeSize());
  m_memory_view = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 DataSynchronizeBufferBase::BufferInfo::
displacement(Int32 rank_index) const
{
  return m_displacements[rank_index][0];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 DataSynchronizeBufferBase::BufferInfo::
localBufferSize(Int32 rank_index) const
{
  return m_local_buffer_size[rank_index];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MutableMemoryView DataSynchronizeBufferBase::BufferInfo::
localBuffer(Int32 rank_index) const
{
  std::byte* data = m_memory_view.data();
  data += m_displacements[rank_index][0];
  const Int64 nb_byte = m_local_buffer_size[rank_index];
  return makeMutableMemoryView(data, 1, nb_byte);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MutableMemoryView DataSynchronizeBufferBase::BufferInfo::
dataLocalBuffer(Int32 rank_index, Int32 data_index) const
{
  std::byte* data = m_memory_view.data();
  data += m_displacements[rank_index][data_index];
  const Int32 nb_item = m_buffer_info->nbItem(rank_index);
  return makeMutableMemoryView(data, m_datatype_sizes[data_index], nb_item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> DataSynchronizeBufferBase::BufferInfo::
localIds(Int32 index) const
{
  return m_buffer_info->localIds(index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Initialise les informations du buffer.
 *
 * Calcul l'offset de chaque donnée de chaque rang dans le buffer global.
 *
 * \note \a datatype_sizes est conservé sous forme de vue et ne doit donc pas
 * être modifié et rester valide durant la synchronisation.
 */
void DataSynchronizeBufferBase::BufferInfo::
initialize(ConstArrayView<Int32> datatype_sizes, const DataSynchronizeBufferInfoList* buffer_info)
{
  ARCANE_CHECK_POINTER(buffer_info);
  m_buffer_info = buffer_info;
  m_datatype_sizes = datatype_sizes;
  const Int32 nb_data = datatype_sizes.size();
  const Int32 nb_rank = buffer_info->nbRank();
  m_displacements.resize(nb_rank, nb_data);
  m_local_buffer_size.resize(nb_rank);

  // Calcul l'offset pour chaque donnée de chaque rang
  // en garantissant que l'offset est un multiple de ALIGNMENT_SIZE
  Int64 data_offset = 0;
  m_total_size = 0;
  for (Int32 i = 0; i < nb_rank; ++i) {
    const Int32 nb_item = buffer_info->nbItem(i);
    Int64 local_buf_nb_byte = 0;
    for (Int32 d = 0; d < nb_data; ++d) {
      // Taille nécessaire pour la donnée \a d pour le rang \a i
      // On fait un padding sur cette taille pour avoir
      // un alignment spécifique.
      const Int64 nb_byte = _applyPadding(nb_item * datatype_sizes[d]);
      m_displacements[i][d] = data_offset;
      local_buf_nb_byte += nb_byte;
      data_offset += nb_byte;
    }
    m_local_buffer_size[i] = local_buf_nb_byte;
    m_total_size += local_buf_nb_byte;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataSynchronizeBufferBase::
DataSynchronizeBufferBase(DataSynchronizeInfo* sync_info, Ref<IBufferCopier> copier)
: m_sync_info(sync_info)
, m_buffer_copier(copier)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 DataSynchronizeBufferBase::
targetRank(Int32 index) const
{
  return m_sync_info->targetRank(index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DataSynchronizeBufferBase::
barrier()
{
  m_buffer_copier->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul et alloue les tampons nécessaires aux envois et réceptions
 * pour les synchronisations des variables 1D.
 */
void DataSynchronizeBufferBase::
_compute(ConstArrayView<Int32> datatype_sizes)
{
  m_nb_rank = m_sync_info->size();

  m_ghost_buffer_info.initialize(datatype_sizes, &m_sync_info->receiveInfo());
  m_share_buffer_info.initialize(datatype_sizes, &m_sync_info->sendInfo());
  m_compare_sync_buffer_info.initialize(datatype_sizes, &m_sync_info->receiveInfo());

  _allocateBuffers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul et alloue les tampons nécessaires aux envois et réceptions
 * pour les synchronisations des variables 1D.
 *
 * Il faut avoir appelé _compute() avant pour calculer les tailles et offset
 * pour chaque buffer.
 *
 * \todo: ne pas converver les tampons pour chaque type de donnée des variables
 * car leur conservation est couteuse en terme de memoire.
 */
void DataSynchronizeBufferBase::
_allocateBuffers()
{
  const Int64 total_ghost_buffer = m_ghost_buffer_info.totalSize();
  const Int64 total_share_buffer = m_share_buffer_info.totalSize();
  Int64 total_compare_buffer = 0;
  if (m_is_compare_sync_values)
    total_compare_buffer = m_compare_sync_buffer_info.totalSize();

  Int64 total_size = total_ghost_buffer + total_share_buffer + total_compare_buffer;
  m_memory->resize(total_size);

  Int64 share_offset = total_ghost_buffer;
  Int64 check_sync_offset = share_offset + total_share_buffer;

  Span<std::byte> buffer_span = m_memory->bytes();
  auto s1 = buffer_span.subspan(0, share_offset);
  m_ghost_buffer_info.setGlobalBuffer(makeMutableMemoryView(s1.data(), 1, total_ghost_buffer));
  auto s2 = buffer_span.subspan(share_offset, total_share_buffer);
  m_share_buffer_info.setGlobalBuffer(makeMutableMemoryView(s2.data(), 1, total_share_buffer));
  if (m_is_compare_sync_values) {
    auto s3 = buffer_span.subspan(check_sync_offset, total_ghost_buffer);
    m_compare_sync_buffer_info.setGlobalBuffer(makeMutableMemoryView(s3.data(), 1, total_ghost_buffer));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SingleDataSynchronizeBuffer::
copyReceiveAsync(Int32 index)
{
  m_ghost_buffer_info.checkValid();

  MutableMemoryView var_values = dataView();
  ConstArrayView<Int32> indexes = m_ghost_buffer_info.localIds(index);
  ConstMemoryView local_buffer = m_ghost_buffer_info.dataLocalBuffer(index, 0);

  m_buffer_copier->copyFromBufferAsync(indexes, local_buffer, var_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SingleDataSynchronizeBuffer::
copySendAsync(Int32 index)
{
  m_share_buffer_info.checkValid();

  ConstMemoryView var_values = dataView();
  ConstArrayView<Int32> indexes = m_share_buffer_info.localIds(index);
  MutableMemoryView local_buffer = m_share_buffer_info.dataLocalBuffer(index, 0);
  m_buffer_copier->copyToBufferAsync(indexes, local_buffer, var_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SingleDataSynchronizeBuffer::
prepareSynchronize(bool is_compare_sync)
{
  m_is_compare_sync_values = is_compare_sync;

  _compute(m_datatype_sizes.view());

  if (is_compare_sync) {
    // Recopie dans le buffer de vérification les valeurs actuelles des mailles
    // fantômes.
    MutableMemoryView var_values = dataView();
    Int32 nb_rank = nbRank();
    for (Int32 i = 0; i < nb_rank; ++i) {
      ConstArrayView<Int32> indexes = m_compare_sync_buffer_info.localIds(i);
      MutableMemoryView local_buffer = m_compare_sync_buffer_info.dataLocalBuffer(i, 0);
      m_buffer_copier->copyToBufferAsync(indexes, local_buffer, var_values);
    }
    // Normalement pas besoin de faire une barrière, car ensuite il y aura les
    // envois sur la même \a queue et ensuite une barrière.
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compare les valeurs avant/après synchronisation.
 *
 * Il suffit de comparer bit à bit le buffer de réception avec celui
 * contenant les valeurs avant la synchronisation (m_check_sync_buffer).
 *
 * \retval \a vrai s'il y a des différences.
 */
DataSynchronizeResult SingleDataSynchronizeBuffer::
finalizeSynchronize()
{
  if (!m_is_compare_sync_values)
    return {};
  ConstMemoryView reference_buffer = m_compare_sync_buffer_info.globalBuffer();
  ConstMemoryView receive_buffer = m_ghost_buffer_info.globalBuffer();
  Span<const std::byte> reference_bytes = reference_buffer.bytes();
  Span<const std::byte> receive_bytes = receive_buffer.bytes();
  Int64 reference_size = reference_bytes.size();
  Int64 receive_size = receive_bytes.size();
  if (reference_size != receive_size)
    ARCANE_FATAL("Incoherent buffer size ref={0} receive={1}", reference_size, receive_size);
  // TODO: gérer le cas où la mémoire est sur le device

  DataSynchronizeResult result;
  bool is_same = std::memcmp(reference_bytes.data(), receive_bytes.data(), reference_size) == 0;
  result.setCompareStatus(is_same ? eDataSynchronizeCompareStatus::Same : eDataSynchronizeCompareStatus::Different);
  return result;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Les comparaisons ne sont pas supportées si on utilise les synchronisations
 * multiples.
 */
void MultiDataSynchronizeBuffer::
prepareSynchronize([[maybe_unused]] bool is_compare_sync)
{
  _compute(m_datatype_sizes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiDataSynchronizeBuffer::
copyReceiveAsync(Int32 rank_index)
{
  IBufferCopier* copier = m_buffer_copier.get();
  m_ghost_buffer_info.checkValid();

  ConstArrayView<Int32> local_ids = m_ghost_buffer_info.localIds(rank_index);
  Int32 data_index = 0;
  for (MutableMemoryView var_values : m_data_views) {
    ConstMemoryView local_buffer = m_ghost_buffer_info.dataLocalBuffer(rank_index, data_index);
    _checkAlignment(local_buffer.data());
    if (!local_buffer.bytes().empty())
      copier->copyFromBufferAsync(local_ids, local_buffer, var_values);
    ++data_index;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiDataSynchronizeBuffer::
copySendAsync(Int32 rank_index)
{
  IBufferCopier* copier = m_buffer_copier.get();
  m_ghost_buffer_info.checkValid();

  ConstArrayView<Int32> local_ids = m_share_buffer_info.localIds(rank_index);
  Int32 data_index = 0;
  for (ConstMemoryView var_values : m_data_views) {
    MutableMemoryView local_buffer = m_share_buffer_info.dataLocalBuffer(rank_index, data_index);
    _checkAlignment(local_buffer.data());
    if (!local_buffer.bytes().empty())
      copier->copyToBufferAsync(local_ids, local_buffer, var_values);
    ++data_index;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
