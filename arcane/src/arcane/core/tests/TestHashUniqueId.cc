// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/FixedArray.h"

#include "arcane/core/MeshUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

namespace
{
inline void _doCheckHash(const char* name,ConstArrayView<Int64> ids)
{
  Int64 hash_uid = MeshUtils::generateHashUniqueId(ids);
  std::cout << name << " = " << hash_uid << "\n";
  if (ids.size()>0) {
    ASSERT_TRUE(hash_uid > 0);
  }
}

}
TEST(ArcaneMeshUtils, HashUniqueId)
{
  FixedArray<Int64,0> nodes0;
  FixedArray<Int64,1> nodes1( { 25 });
  FixedArray<Int64,2> nodes2( { 25, 37 });
  FixedArray<Int64,3> nodes3 ({ 25, 37, 48 });
  _doCheckHash("X0",nodes0.view());
  _doCheckHash("X1",nodes1.view());
  _doCheckHash("X2",nodes2.view());
  _doCheckHash("X3",nodes3.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
