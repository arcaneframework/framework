// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItem.h                                                   (C) 2000-2012 */
/*                                                                           */
/* Include utilisateur regroupant les fonctionnalités AnyItem                */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ANYITEM_ANYITEM_H
#define ARCANE_ANYITEM_ANYITEM_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/anyitem/AnyItemGroup.h"
#include "arcane/core/anyitem/AnyItemFamilyObserver.h"
#include "arcane/core/anyitem/AnyItemFamily.h"
#include "arcane/core/anyitem/AnyItemUserGroup.h"
#include "arcane/core/anyitem/AnyItemVariable.h"
#include "arcane/core/anyitem/AnyItemVariableArray.h"
#include "arcane/core/anyitem/AnyItemLinkFamily.h"
#include "arcane/core/anyitem/AnyItemLinkVariable.h"
#include "arcane/core/anyitem/AnyItemLinkVariableArray.h"
#include "arcane/core/anyitem/AnyItemArray.h"
#include "arcane/core/anyitem/AnyItemArray2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ENUMERATE_ANY_ITEM(name, group)                           \
  for(AnyItem::Group::Enumerator __e((group).enumerator()); __e.hasNext(); ++__e) \
    for(AnyItem::Group::BlockItemEnumerator name(__e.enumerator()); name.hasNext(); ++name)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ENUMERATE_ANY_ITEM_LINK(name, container)             \
  for(AnyItem::LinkFamily::Enumerator name((container).enumerator()); \
      name.hasNext(); ++name )

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
     
#endif /* ARCANE_ANYITEM_ANYITEM_H */
