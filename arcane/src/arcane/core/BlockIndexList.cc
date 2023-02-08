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

  Int32 new_size = m_indexes.size() + m_blocks_index_and_offset.size();
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
  m_blocks_index_and_offset.clear();
  m_original_size = 0;
  m_block_size = 0;
  m_nb_block = 0;
  m_last_block_size = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BlockIndexList::
_setBlockIndexAndOffset(Int32 block, Int32 index, Int32 offset)
{
  m_blocks_index_and_offset[block * 2] = index * m_block_size;
  m_blocks_index_and_offset[(block * 2) + 1] = offset;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BlockIndexList::
_setNbBlock(Int32 nb_block)
{
  m_blocks_index_and_offset.resize(nb_block * 2);
  m_nb_block = nb_block;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 BlockIndexList::
_currentIndexPosition() const
{
  return m_indexes.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BlockIndexList::
_addBlockInfo(const Int32* data, Int16 size)
{
  m_indexes.addRange(ConstArrayView<Int32>(size, data));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BlockIndexListBuilder::
build(BlockIndexList& block_index_list, SmallSpan<const Int32> indexes, const String& name)
{
  block_index_list.reset();

  bool is_verbose = m_is_verbose;

  const Int16 block_size = (m_block_size > 0) ? m_block_size : 32;
  const Int32 original_size = indexes.size();
  const Int32 nb_fixed_block = original_size / block_size;
  const Int16 remaining_size = static_cast<Int16>(original_size % block_size);
  Int32 nb_block = nb_fixed_block;
  Int16 last_block_size = block_size;
  if (remaining_size != 0) {
    ++nb_block;
    last_block_size = remaining_size;
  }

  Int32 nb_contigu = 0;
  std::unordered_map<std::size_t, Int32> block_indexes;
  std::hash<Int32> hasher;
  std::ostringstream o;
  block_index_list._setNbBlock(nb_block);
  block_index_list.m_original_size = original_size;
  block_index_list.m_block_size = block_size;
  block_index_list.m_last_block_size = last_block_size;

  UniqueArray<Int32> block_index_in_original_array;
  block_index_in_original_array.reserve(nb_block);

  for (Int32 i = 0; i < nb_fixed_block; ++i) {
    bool is_contigu = true;
    Int32 iter_index = i * block_size;
    Int32 first_value = indexes[iter_index];
    size_t hash = hasher(0);

    // TODO: faire une spécialisation en fonction de la taille de bloc.
    for (Int32 z = 1; z < block_size; ++z) {
      Int32 diff = indexes[iter_index + z] - first_value;
      size_t hash2 = hasher(diff);
      hash ^= hash2 + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      if (indexes[iter_index + z] != first_value + z) {
        is_contigu = false;
      }
    }

    auto idx = block_indexes.find(hash);
    Int32 block_index = -1;
    if (idx == block_indexes.end()) {
      // Nouveau bloc.
      block_index = static_cast<Int32>(block_indexes.size());
      block_indexes.insert(std::make_pair(hash, block_index));
      block_index_in_original_array.add(iter_index);
    }
    else
      block_index = idx->second;
    block_index_list._setBlockIndexAndOffset(i, block_index, first_value);
    if (is_contigu)
      ++nb_contigu;

    if (is_verbose) {
      o << "\nBlock i=" << std::setw(5) << i;
      for (Int32 z = 1; z < block_size; ++z) {
        Int32 diff = indexes[iter_index + z] - first_value;
        o << " " << std::setw(4) << diff;
      }
      o << " H=" << std::hex << hash << std::setbase(0);
    }
  }

  // Gère l'éventuel dernier bloc.
  if (remaining_size != 0) {
    Int32 iter_index = nb_fixed_block * block_size;
    Int32 first_value = indexes[iter_index];
    Int32 block_index = static_cast<Int32>(block_indexes.size());
    block_index_in_original_array.add(iter_index);
    block_index_list._setBlockIndexAndOffset(nb_fixed_block, block_index, first_value);
  }

  // Maintenant qu'on connait le nombre de blocs communs, alloue le tableau
  // des blocs compressés et recopie les valeurs correspondantes
  {
    Int32 local_block_values[BlockIndex::MAX_BLOCK_SIZE];
    local_block_values[0] = 0;
    const Int32 nb_reduced_block = static_cast<Int32>(block_indexes.size());
    block_index_list.m_indexes.reserve((block_size * nb_reduced_block) + remaining_size);
    for (Int32 i = 0; i < nb_reduced_block; ++i) {
      Int32 pos = block_index_in_original_array[i];
      Int32 first_value = indexes[pos];
      for (Int32 z = 1; z < block_size; ++z) {
        Int32 diff = indexes[pos + z] - first_value;
        local_block_values[z] = diff;
      }
      block_index_list.m_indexes.addRange(ConstArrayView<Int32>(block_size, local_block_values));
    }
    if (remaining_size != 0) {
      Int32 pos = nb_fixed_block * block_size;
      Int32 first_value = indexes[pos];
      // TODO: regarder pour faire un padding jusqu'à la fin du bloc avec la dernière
      // valeur (comme ce qui est fait pour les ItemGroup)
      for (Int32 z = 1; z < remaining_size; ++z) {
        Int32 diff = indexes[pos + z] - first_value;
        local_block_values[z] = diff;
      }
      block_index_list.m_indexes.addRange(ConstArrayView<Int32>(remaining_size, local_block_values));
    }
  }

  if (is_verbose)
    info() << o.str();
  info() << "Group Name=" << name << " original_size = " << original_size << " nb_block = " << nb_block
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

void BlockIndexListBuilder::
setBlockSizeAsPowerOfTwo(Int32 v)
{
  if (v < 0)
    _throwInvalidBlockSize(v);
  Int32 block_size = 1 << v;
  if (block_size > BlockIndex::MAX_BLOCK_SIZE)
    _throwInvalidBlockSize(block_size);
  m_block_size = static_cast<Int16>(block_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BlockIndexListBuilder::
_throwInvalidBlockSize(Int32 block_size)
{
  ARCANE_FATAL("Bad value for block size v={0} min=1 max={1}",
               block_size, BlockIndex::MAX_BLOCK_SIZE);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
