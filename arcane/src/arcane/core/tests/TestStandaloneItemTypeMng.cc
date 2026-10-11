// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/core/ItemTypeMng.h"
#include "arcane/core/ItemTypeInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

TEST(ArcaneMeshUtils, ItemTypeMng)
{
  auto x = ItemTypeMng::createStandalone();
  ConstArrayView<ItemTypeInfo*> types = x->types();
  std::cout << "NB_TYPE=" << types.size() << "\n";
  for (ItemTypeInfo* iti : types) {
    std::cout << "TYPE id=" << iti->typeId() << " name=" << iti->typeName() << "\n";
  }
  ASSERT_EQ(types.size(),ItemTypeMng::nbBuiltInItemType());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
