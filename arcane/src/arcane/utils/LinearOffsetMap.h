// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LinearOffsetMap.h                                           (C) 2000-2024 */
/*                                                                           */
/* List of linear offsets.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_LINEAROFFSETMAP_H
#define ARCANE_UTILS_LINEAROFFSETMAP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief List of linear offsets.
 *
 * `DataType` must be `Int32` or `Int64`.
 *
 * \warning Experimental class. Do not use outside of Arcane.
 */
template <typename DataType>
class LinearOffsetMap
{
 public:

  static_assert(std::is_same_v<DataType, Int32> || std::is_same_v<DataType, Int64>);

 public:

  //! Adds an offset \a offset of size \a size
  ARCANE_UTILS_EXPORT void add(DataType size, DataType offset);

  /*!
   * \brief Retrieves a sufficient offset for an element of size \a size.
   *
   * Returns a negative value if no offset is available. If an offset
   * is available, it returns its value. The found offset is removed from the list
   * and an offset is added for the remaining size if it is not zero:
   * if the found offset is `offset` and the associated size is `offset_size`,
   * call `add(offset_size - size, offset + size)`.
   */
  ARCANE_UTILS_EXPORT DataType getAndRemoveOffset(DataType size);

  //! Number of elements in the table.
  ARCANE_UTILS_EXPORT Int32 size() const;

 private:

  std::multimap<DataType, DataType> m_offset_map;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
