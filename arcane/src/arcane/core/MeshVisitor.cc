// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshUtils.cc                                                (C) 2000-2023 */
/*                                                                           */
/* Visiteurs divers sur les entités du maillage.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Enumerator.h"
#include "arcane/utils/Collection.h"

#include "arcane/IItemFamily.h"
#include "arcane/IMesh.h"
#include "arcane/ItemGroup.h"
#include "arcane/MeshUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
visitGroups(IItemFamily* family, IFunctorWithArgumentT<ItemGroup&>* functor)
{
  for (ItemGroupCollection::Enumerator i(family->groups()); ++i;) {
    ItemGroup& group = *i;
    functor->executeFunctor(group);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
visitGroups(IMesh* mesh, IFunctorWithArgumentT<ItemGroup&>* functor)
{
  for (IItemFamilyCollection::Enumerator ifamily(mesh->itemFamilies()); ++ifamily;) {
    IItemFamily* family = *ifamily;
    for (ItemGroupCollection::Enumerator i(family->groups()); ++i;) {
      ItemGroup& group = *i;
      functor->executeFunctor(group);
    }
  }
}

static void test_compile()
{
  IItemFamily* f = nullptr;
  auto xx = [](const ItemGroup& x) { std::cout << "name=" << x.name(); };
  MeshUtils::visitGroups(f, xx);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
