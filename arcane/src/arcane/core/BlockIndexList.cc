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

#include <iomanip>
#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BlockIndexListBuilder::
build(SmallSpan<const Int32> indexes,const String& name)
{
  bool is_verbose = m_is_verbose;

  const Int32 block_size = (m_block_size>0) ? m_block_size : 32;

  Int32 size = indexes.size();
  //Int32 nb_block = (size + (block_size - 1)) / block_size;
  // Pour ce test ne traite pas l'éventuel dernier bloc.
  Int32 nb_block = size / block_size;

  Int32 nb_contigu = 0;
  std::unordered_map<std::size_t, Int32> block_occurences;
  std::hash<Int32> hasher;
  std::ostringstream o;
  for (Int32 i = 0; i < nb_block; ++i) {
    bool is_contigu = true;
    Int32 iter_index = i * block_size;
    Int32 first_value = indexes[iter_index];
    size_t hash = hasher(0);
    if (is_verbose)
      o << "\nBlock i=" << std::setw(5) << i;
    for (Int32 z = 1; z < block_size; ++z) {
      Int32 diff = indexes[iter_index + z] - first_value;
      size_t hash2 = hasher(diff);
      hash ^= hash2 + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      if (is_verbose)
        o << " " << std::setw(4) << diff;
      if (indexes[iter_index + z] != first_value + z) {
        is_contigu = false;
      }
    }
    ++block_occurences[hash];
    if (is_verbose)
      o << " H=" << std::hex << hash << std::setbase(0);
    if (is_contigu)
      ++nb_contigu;
  }

  if (is_verbose)
    info() << o.str();
  info() << "Group Name=" << name << " size = " << size << " nb_block = " << nb_block
         << " nb_contigu=" << nb_contigu
         << " reduced_nb_block=" << block_occurences.size();
  // TODO: gérer le dernier sous-bloc.
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
