// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemInternalSortFunction.h                                 (C) 2000-2025 */
/*                                                                           */
/* Interface of an entity sorting function.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMINTERNALSORTFUNCTION_H
#define ARCANE_CORE_IITEMINTERNALSORTFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Mesh
 * \brief Interface of an entity sorting function.
 *
 * This class is used to sort entities.
 * This is done when calling the sortItems() method.
 *
 * To simplify sorting, it is preferable to use the
 * ItemInternalSortFunction class by specifying the comparison function.
 *
 */
class IItemInternalSortFunction
{
 public:

  virtual ~IItemInternalSortFunction() = default; //!< Releases resources

 public:

  /*!
   * \brief Name of the sorting function.
   *
   * Names starting with 'Arcane' are reserved and must not be
   * used.
   */
  virtual const String& name() const = 0;

  /*!
   * \brief Sorts the entities in the array \a items.
   */
  virtual void sortItems(ItemInternalMutableArrayView items) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
