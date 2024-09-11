﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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
  // Pour accès aux méthodes qui utilisent ItemInternal.
  friend class DynamicMeshKindInfos;

 private:

  using BaseClass = HashTableMapT<Int64, ItemInternal*>;
  using BaseData = BaseClass::Data;

 public:

  using Data ARCANE_DEPRECATED_REASON("Data type is internal to Arcane") = BaseClass::Data;

 public:

  using BaseClass::add;
  using BaseClass::clear;
  using BaseClass::count;
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
    ConstArrayView<BaseData*> b = BaseClass::buckets();
    for (Int32 k = 0, n = b.size(); k < n; ++k) {
      BaseData* nbid = b[k];
      for (; nbid; nbid = nbid->next()) {
        lambda(Arcane::impl::ItemBase(nbid->value()));
      }
    }
  }

 public:

  //! Retourne l'entité associée à \a key si trouvé ou l'entité nulle sinon
  impl::ItemBase tryFind(Int64 key) const
  {
    const BaseData* d = BaseClass::lookup(key);
    return (d ? impl::ItemBase(d->value()) : impl::ItemBase{});
  }
  //! Retourne le localId() associé à \a key si trouvé ou NULL_ITEM_LOCAL_ID sinon aucun
  Int32 tryFindLocalId(Int64 key) const
  {
    const BaseData* d = BaseClass::lookup(key);
    return (d ? d->value()->localId() : NULL_ITEM_LOCAL_ID);
  }

 private:

  //! Retourne l'entité associée à \a key si trouvé ou nullptr sinon
  ItemInternal* tryFindItemInternal(Int64 key) const
  {
    const BaseData* d = BaseClass::lookup(key);
    return (d ? d->value() : nullptr);
  }

 public:

  ARCANE_DEPRECATED_REASON("This method is internal to Arcane")
  Data* lookup(Int64 key)
  {
    return BaseClass::lookup(key);
  }

  ARCANE_DEPRECATED_REASON("This method is internal to Arcane")
  const Data* lookup(Int64 key) const
  {
    return BaseClass::lookup(key);
  }

  ARCANE_DEPRECATED_REASON("This method is internal to Arcane")
  ConstArrayView<BaseData*> buckets() const { return BaseClass::buckets(); }

 private:

  /*!
   * \brief Change la valeurs des localId(0.
   *
   * Cette méthode ne doit être appelée que par DynamicMeshKindInfos.
   */
  void changeLocalIds(ArrayView<ItemInternal*> items_internal,
                      ConstArrayView<Int32> old_to_new_local_ids);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Macro pour itérer sur les valeurs d'un ItemInternalMap
#define ENUMERATE_ITEM_INTERNAL_MAP_DATA(iter,item_list) \
for( auto __i__##iter : item_list .buckets() ) \
    for (auto* iter = __i__##iter; iter; iter = iter->next())

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
