// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternalSortFunction.h                                  (C) 2000-2025 */
/*                                                                           */
/* Fonction de tri des entités.                                              */
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
 * \brief Interface d'une fonction de tri des entités.
 *
 * Cette classe est utilisée pour trier des entités.
 * Il faut spécifier comme paramètre template un functor
 * ayant le prototype suivant:
 * \code
 * bool operator()(const ItemInternal* item1,const ItemInternal* item2) const
 * \endcode
 * et qui retourne \a true si \a item1 est avant \a item2.
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
   * \brief Trie les entités du tableau \a items.
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
