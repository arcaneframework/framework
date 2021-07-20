// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/NumArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(NumArray3,Misc)
{
  using namespace Arcane;
  int nb_x = 3;
  int nb_y = 4;
  int nb_z = 5;

  NumArray<Int64,3> v(nb_x,nb_y,nb_z);
  {
    for( Int64 x=0, xn=v.dim1Size(); x<xn; ++x ){
      for( Int64 y=0, yn=v.dim2Size(); y<yn; ++y ){
        for( Int64 z=0, zn=v.dim3Size(); z<zn; ++z ){
          v.s(x,y,z) = x+y+z+1;
          v.s({x,y,z}) = x+y+z+1;
        }
      }
    }
  }

  for( Int64 x=0, xn=v.dim1Size(); x<xn; ++x ){
    for( Int64 y=0, yn=v.dim2Size(); y<yn; ++y ){
      for( Int64 z=0, zn=v.dim3Size(); z<zn; ++z ){
        Int64 val1 = v(x,y,z);
        Int64 val2 = v({x,y,z});
        ASSERT_TRUE(val1==val2) << "Difference values v1=" << val1 << " v2=" << val2;
      }
    }
  }
}
