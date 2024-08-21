// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HashTable.cc                                                (C) 2000-2024 */
/*                                                                           */
/* Table de hachage.                                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/LinearOffsetMap.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void LinearOffsetMap<DataType>::
add(DataType size, DataType offset)
{
  std::cout << "ADD size=" << size << " offset=" << offset << "\n";
  m_offset_map.insert(std::make_pair(size, offset));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> DataType LinearOffsetMap<DataType>::
getAndRemoveOffset(DataType size)
{
  auto x = m_offset_map.lower_bound(size);
  std::cout << "TRY_FIND size=" << size << " n=" << m_offset_map.size() << "\n";
  if (x == m_offset_map.end())
    return (-1);
  DataType offset = x->second;
  DataType offset_size = x->first;
  std::cout << "FOUND size=" << size << " found_offset=" << offset
            << " offset_size=" << offset_size << "\n";
  m_offset_map.erase(x);
  DataType remaining_size = offset_size - size;
  if (remaining_size != 0)
    add(remaining_size, offset + size);
  return offset;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> Int32 LinearOffsetMap<DataType>::
size() const
{
  return static_cast<Int32>(m_offset_map.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class LinearOffsetMap<Int32>;
template class LinearOffsetMap<Int64>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
