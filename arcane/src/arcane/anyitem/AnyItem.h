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

#include "arcane/anyitem/AnyItemGroup.h"
#include "arcane/anyitem/AnyItemFamilyObserver.h"
#include "arcane/anyitem/AnyItemFamily.h"
#include "arcane/anyitem/AnyItemUserGroup.h"
#include "arcane/anyitem/AnyItemVariable.h"
#include "arcane/anyitem/AnyItemVariableArray.h"
#include "arcane/anyitem/AnyItemLinkFamily.h"
#include "arcane/anyitem/AnyItemLinkVariable.h"
#include "arcane/anyitem/AnyItemLinkVariableArray.h"
#include "arcane/anyitem/AnyItemArray.h"
#include "arcane/anyitem/AnyItemArray2.h"

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
