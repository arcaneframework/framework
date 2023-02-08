// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BlockIndexList.cc                                           (C) 2000-2023 */
/*                                                                           */
/* Classe gérant un tableau d'indices sous la forme d'une liste de blocs.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/BlockIndexList.h"

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/String.h"
#include "arcane/utils/FatalErrorException.h"

#include <iomanip>
#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real BlockIndexList::
memoryRatio() const
{
  if (m_original_size == 0)
    return 0.0;

  Int32 new_size = m_indexes.size() + m_block_indexes.size() + m_block_offsets.size();
  return (static_cast<Real>(new_size) / static_cast<Real>(m_original_size));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BlockIndexList::
fillArray(Array<Int32>& v)
{
  for (Int32 i = 0, n = m_nb_block; i < n; ++i) {
    BlockIndex bi = block(i);
    for (Int32 z = 0, nb_z = bi.size(); z < nb_z; ++z) {
      v.add(bi[z]);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BlockIndexList::
reset()
{
  m_indexes.clear();
  m_block_indexes.clear();
  m_block_offsets.clear();
  m_original_size = 0;
  m_block_size = 0;
  m_nb_block = 0;
  m_last_block_size = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BlockIndexListBuilder::
build(BlockIndexList& block_index_list, SmallSpan<const Int32> indexes, const String& name)
{
  block_index_list.reset();

  bool is_verbose = m_is_verbose;

  const Int32 block_size = (m_block_size > 0) ? m_block_size : 32;
  constexpr Int32 MAX_BLOCK_SIZE = 512;
  if (block_size > MAX_BLOCK_SIZE)
    ARCANE_FATAL("Bad value for block size v={0} max={1}", block_size, MAX_BLOCK_SIZE);
  const Int32 size = indexes.size();
  //Int32 nb_block = (size + (block_size - 1)) / block_size;
  // Pour ce test ne traite pas l'éventuel dernier bloc.
  const Int32 nb_fixed_block = size / block_size;
  const Int32 remaining_size = (size % block_size);
  Int32 nb_block = nb_fixed_block;
  Int32 last_block_size = block_size;
  if (remaining_size != 0) {
    ++nb_block;
    last_block_size = remaining_size;
  }

  Int32 nb_contigu = 0;
  std::unordered_map<std::size_t, Int32> block_indexes;
  std::hash<Int32> hasher;
  std::ostringstream o;
  block_index_list.m_block_indexes.resize(nb_block);
  block_index_list.m_block_offsets.resize(nb_block);
  block_index_list.m_original_size = size;
  block_index_list.m_block_size = block_size;
  block_index_list.m_nb_block = nb_block;
  block_index_list.m_last_block_size = last_block_size;
  Int32 local_block_values[MAX_BLOCK_SIZE];
  local_block_values[0] = 0;
  for (Int32 i = 0; i < nb_fixed_block; ++i) {
    bool is_contigu = true;
    Int32 iter_index = i * block_size;
    Int32 first_value = indexes[iter_index];
    size_t hash = hasher(0);
    if (is_verbose)
      o << "\nBlock i=" << std::setw(5) << i;
    // TODO: faire une spécialisation en fonction de la taille de bloc.
    for (Int32 z = 1; z < block_size; ++z) {
      Int32 diff = indexes[iter_index + z] - first_value;
      local_block_values[z] = diff;
      size_t hash2 = hasher(diff);
      hash ^= hash2 + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      if (is_verbose)
        o << " " << std::setw(4) << diff;
      if (indexes[iter_index + z] != first_value + z) {
        is_contigu = false;
      }
    }
    auto idx = block_indexes.find(hash);
    Int32 block_index = -1;
    if (idx == block_indexes.end()) {
      // Nouveau bloc.
      block_index = block_index_list.m_indexes.size();
      block_index_list.m_indexes.addRange(ConstArrayView<Int32>(block_size, local_block_values));
      block_indexes.insert(std::make_pair(hash, block_index));
    }
    else
      block_index = idx->second;
    block_index_list.m_block_indexes[i] = block_index;
    block_index_list.m_block_offsets[i] = first_value;
    if (is_verbose)
      o << " H=" << std::hex << hash << std::setbase(0);
    if (is_contigu)
      ++nb_contigu;
  }

  // Gère l'éventuel dernier bloc.
  if (remaining_size != 0) {
    Int32 iter_index = nb_fixed_block * block_size;
    Int32 first_value = indexes[iter_index];
    for (Int32 z = 1; z < remaining_size; ++z) {
      Int32 diff = indexes[iter_index + z] - first_value;
      local_block_values[z] = diff;
    }
    Int32 block_index = block_index_list.m_indexes.size();
    block_index_list.m_indexes.addRange(ConstArrayView<Int32>(remaining_size, local_block_values));
    block_index_list.m_block_indexes[nb_fixed_block] = block_index;
    block_index_list.m_block_offsets[nb_fixed_block] = first_value;
  }

  if (is_verbose)
    info() << o.str();
  info() << "Group Name=" << name << " size = " << size << " nb_block = " << nb_block
         << " nb_contigu=" << nb_contigu
         << " reduced_nb_block=" << block_indexes.size()
         << " ratio=" << block_index_list.memoryRatio();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BlockIndexListBuilder::
BlockIndexListBuilder(ITraceMng* tm)
: TraceAccessor(tm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
