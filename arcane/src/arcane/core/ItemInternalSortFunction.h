// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternalSortFunction.h                                  (C) 2000-2025 */
/*                                                                           */
/* Entity sorting function.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMINTERNALSORTFUNCTION_H
#define ARCANE_CORE_ITEMINTERNALSORTFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/core/IItemInternalSortFunction.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Mesh
 * \brief Interface for an entity sorting function.
 *
 * This class is used to sort entities.
 * A functor must be specified as a template parameter
 * having the following prototype:
 * \code
 * bool operator()(const ItemInternal* item1,const ItemInternal* item2) const
 * \endcode
 * and which returns true if item1 comes before item2.
 */
template <typename SortFunction>
class ItemInternalSortFunction
: public IItemInternalSortFunction
{
 public:

  explicit ItemInternalSortFunction(const String& name)
  : m_name(name)
  {}

 public:

  const String& name() const override { return m_name; }

 public:

  /*!
   * \brief Sorts the entities in the array \a items.
   */
  void sortItems(ItemInternalMutableArrayView items) override
  {
    std::sort(std::begin(items), std::end(items), SortFunction());
  }

 private:

  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
