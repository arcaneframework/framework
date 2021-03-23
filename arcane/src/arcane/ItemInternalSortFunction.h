// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternalSortFunction.h                                  (C) 2000-2008 */
/*                                                                           */
/* Fonction de tri des entités.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMINTERNALSORTFUNCTION_H
#define ARCANE_ITEMINTERNALSORTFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/IItemInternalSortFunction.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
template<typename SortFunction>
class ItemInternalSortFunction
: public IItemInternalSortFunction
{
 public:

  ItemInternalSortFunction(const String& name)
  : m_name(name){}
  virtual ~ItemInternalSortFunction() {} //<! Libère les ressources

 public:

  virtual const String& name() const
  {
    return m_name;
  }

 public:

  /*!
   * \brief Trie les entités du tableau \a items.
   */
  virtual void sortItems(ItemInternalMutableArrayView items)
  {
    std::sort(std::begin(items),std::end(items),SortFunction());
  }

 private:

  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
