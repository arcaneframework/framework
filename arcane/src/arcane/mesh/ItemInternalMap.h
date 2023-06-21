// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternalMap.h                                           (C) 2000-2023 */
/*                                                                           */
/* Tableau associatif de ItemInternal.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMINTERNALMAP_H
#define ARCANE_MESH_ITEMINTERNALMAP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashTableMap.h"

#include "arcane/mesh/MeshGlobal.h"

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
: public HashTableMapT<Int64,ItemInternal*>
{
 private:
  typedef HashTableMapT<Int64,ItemInternal*> BaseClass;
 public:
  ItemInternalMap();
 public:
  void notifyUniqueIdsChanged();
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
