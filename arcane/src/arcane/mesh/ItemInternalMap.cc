// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternalMap.cc                                          (C) 2000-2021 */
/*                                                                           */
/* Tableau associatif de ItemInternal.                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/ItemInternalMap.h"

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/ItemInternal.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternalMap::
ItemInternalMap()
: BaseClass(5000,false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternalMap::
notifyUniqueIdsChanged()
{
  BaseClass& c = *this;
  // Vérifie qu'on n'a pas deux fois la même clé.
  {
    std::set<Int64> uids;
    ENUMERATE_ITEM_INTERNAL_MAP_DATA(nbid,c){
      Int64 uid = nbid->value()->uniqueId().asInt64();
      if (uids.find(uid)!=uids.end())
        ARCANE_FATAL("Duplicated uniqueId '{0}'",uid);
      uids.insert(uid);
    }
  }

  ENUMERATE_ITEM_INTERNAL_MAP_DATA(nbid,c){
    nbid->setKey(nbid->value()->uniqueId().asInt64());
  }
  this->rehash();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
