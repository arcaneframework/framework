// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItem.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Include utilisateur regroupant les fonctionnalités AnyItem                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ANYITEM_ANYITEM_PRIVATE_H
#define ARCANE_CORE_ANYITEM_ANYITEM_PRIVATE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/anyitem/AnyItemGlobal.h"
#include "arcane/core/ItemGroupImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ANYITEM_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct Private 
{
  struct GroupIndexInfo
  {
    ItemGroupImpl * group;
    Integer group_index;
    Integer local_id_offset;
    bool is_partial;
  };

  struct GroupIndexMapping : public UniqueArray<GroupIndexInfo> {
  public:
    const GroupIndexInfo * findGroupInfo(const ItemGroupImpl * group) const {
      const Array<GroupIndexInfo> & self = *this;
      for(Integer i=0;i<size();++i)
        if (self[i].group == group)
          return &self[i];
      return NULL;
    }
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ANYITEM_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
     
#endif /* ARCANE_ANYITEM_ANYITEM_PRIVATE_H */
