// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternalMap.h                                           (C) 2000-2024 */
/*                                                                           */
/* Tableau associatif de ItemInternal.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMINTERNALMAP_H
#define ARCANE_MESH_ITEMINTERNALMAP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashTableMap.h"

#include "arcane/mesh/MeshGlobal.h"
#include "arcane/core/ItemInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemInternal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tableau associatif de ItemInternal.
 *
 * La clé de ce tableau associatif est le UniqueId des entités.
 * S'il change, il faut appeler notifyUniqueIdsChanged() pour remettre
 * à jour le tableau associatif.
 */
class ItemInternalMap
: private HashTableMapT<Int64, ItemInternal*>
{
 private:

  using BaseClass = HashTableMapT<Int64, ItemInternal*>;

 public:

  using Data = BaseClass::Data;
  using BaseClass::add;
  using BaseClass::buckets;
  using BaseClass::clear;
  using BaseClass::count;
  using BaseClass::lookup;
  using BaseClass::lookupAdd;
  using BaseClass::lookupValue;
  using BaseClass::remove;
  using BaseClass::operator[];
  using BaseClass::hasKey;
  using BaseClass::resize;

 public:

  ItemInternalMap();

 public:

  void notifyUniqueIdsChanged();
  /*!
   * \brief Fonction template pour itérer sur les entités de l'instance.
   *
   * Le type de l'arguments template peut-être n'importe quel type d'entité
   * qui peut être construit à partir d'un impl::ItemBase.
   * \code
   * ItemInternalMap item_map = ...
   * item_map.eachItemBase([&](Item item){
   *   std::cout << "LID=" << item_base.localId() << "\n";
   * });
   * \endcode
   */
  template <class Lambda> void
  eachItem(const Lambda& lambda)
  {
    ConstArrayView<BaseClass::Data*> b = buckets();
    for (Integer k = 0, n = b.size(); k < n; ++k) {
      Data* nbid = b[k];
      for (; nbid; nbid = nbid->next()) {
        lambda(Arcane::impl::ItemBase(nbid->value()));
      }
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Macro pour itérer sur les valeurs d'un ItemInternalMap
#define ENUMERATE_ITEM_INTERNAL_MAP_DATA(iter,item_list) \
for( auto __i__##iter : item_list .buckets() ) \
  for( Arcane::mesh::ItemInternalMap::Data* iter = __i__##iter; iter; iter = iter->next() )

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
