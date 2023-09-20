// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/collections/Array.h"
#include "arccore/collections/Array2.h"
#include "arccore/collections/IMemoryAllocator.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/base/Iterator.h"

#include "TestArrayCommon.h"

using namespace Arccore;
using namespace TestArccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
void
_Add(Array<Real>& v,Integer new_size)
{
  v.resize(new_size);
}
}

namespace
{

template <typename T>
void _dumpArray2(std::ostream& o, const Array2<T>& a)
{
  for (Int32 i = 0; i < a.dim1Size(); ++i)
    for (Int32 j = 0; j < a.dim2Size(); ++j) {
      if (i != 0 || j != 0)
        o << ' ';
      o << "[" << i << "," << j << "]=\"" << a[i][j] << '"';
    }
}
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array2, Misc)
{
  using namespace Arccore;
  PrintableMemoryAllocator printable_allocator2;
  IMemoryAllocator* allocator2 = &printable_allocator2;

  SharedArray2<Int32> sh_a;
  sh_a.resize(3, 2);
  ASSERT_EQ(sh_a.totalNbElement(), 6);
  ASSERT_EQ(sh_a.dim1Size(), 3);
  ASSERT_EQ(sh_a.dim2Size(), 2);

  SharedArray2<Int32> sh_empty;
  {
    SharedArray2<IntSubClass> sh_d;
    sh_d.resize(2, 6);

    SharedArray2<IntSubClass> sh_b;
    sh_b.resize(5, 3);
    for (Int32 i = 0; i < sh_b.dim1Size(); ++i)
      for (Int32 j = 0; j < sh_b.dim2Size(); ++j) {
        sh_b[i][j] = IntSubClass(i + j);
      }
    std::cout << "\nSH_B=";
    _dumpArray2(std::cout, sh_b);
    std::cout << "\n";

    SharedArray2<IntSubClass> sh_c = sh_b;
    sh_d = sh_c;
    std::cout << "\nSH_D=";
    _dumpArray2(std::cout, sh_d);
    std::cout << "\n";
    _checkSameInfoArray2(sh_b, sh_c);
    _checkSameInfoArray2(sh_b, sh_d);
  }
  {
    UniqueArray2<Int32> c;
    c.resize(3, 5);
    Integer nb = 15;
    c.reserve(nb * 2);
    Int64 current_capacity = c.capacity();
    ASSERT_EQ(current_capacity, (nb * 2)) << "Bad capacity (test 1)";
    c.shrink(32);
    ASSERT_EQ(c.capacity(), current_capacity) << "Bad capacity (test 2)";
    c.shrink();
    c.shrink_to_fit();
    ASSERT_EQ(c.capacity(), c.totalNbElement()) << "Bad capacity (test 3)";
    ASSERT_EQ(c[1][2], c(1, 2));
#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
    bool is_ok = c[2, 1] == c(2, 1);
    ASSERT_TRUE(is_ok);
#endif
  }
  {
    UniqueArray2<Int32> c;
    c.resize(2, 1);
    std::cout << "V1=" << c.to1DSpan() << "\n";
    c[0][0] = 2;
    c[1][0] = 3;
    c.resize(2, 2);
    std::cout << "V2=" << c.to1DSpan() << "\n";
    ASSERT_EQ(c[0][0], 2);
    ASSERT_EQ(c[1][0], 3);
    ASSERT_EQ(c[0][1], 0);
    ASSERT_EQ(c[1][1], 0);
    UniqueArray2<Int32> d;
    d.resize(4, 5);
    ASSERT_EQ(d.totalNbElement(), 20);
    ASSERT_EQ(d.dim1Size(), 4);
    ASSERT_EQ(d.dim2Size(), 5);

    d = c;
    ASSERT_EQ(d.totalNbElement(), c.totalNbElement());
    ASSERT_EQ(d.dim1Size(), c.dim1Size());
    ASSERT_EQ(d.dim2Size(), c.dim2Size());

    UniqueArray2<Int32> e(allocator2);
    ASSERT_EQ(e.allocator(), allocator2);
    e.resize(7, 6);
    ASSERT_EQ(e.totalNbElement(), 42);
    ASSERT_EQ(e.dim1Size(), 7);
    ASSERT_EQ(e.dim2Size(), 6);

    {
      e = d;
      ASSERT_EQ(e.allocator(), d.allocator());
      ASSERT_EQ(d.totalNbElement(), e.totalNbElement());
      Int32 dim1_size = e.dim1Size();
      Int32 dim2_size = e.dim2Size();
      ASSERT_EQ(d.dim1Size(), dim1_size);
      ASSERT_EQ(d.dim2Size(), dim2_size);
      e.add(23);
      ASSERT_EQ(e.dim1Size(), dim1_size + 1);
      ASSERT_EQ(e.dim2Size(), dim2_size);
      ASSERT_EQ(e[dim1_size][0], 23);
    }
    {
      UniqueArray2<Int32> f(allocator2);
      UniqueArray2<Int32> g;
      g.resize(4, 3);
      g = f;
      _checkSameInfoArray2(g, f);

      UniqueArray2<Int32> h(f);
      _checkSameInfoArray2(h, f);

      UniqueArray2<Int32> h2(sh_a);
      _checkSameInfoArray2(h2, sh_a);

      g = sh_a;
      _checkSameInfoArray2(g, sh_a);
    }
  }
  {
    UniqueArray2<Int32> c;
    c.resizeNoInit(2, 1);
    c[0][0] = 1;
    c[1][0] = 2;
    c.resizeNoInit(2, 2);
    c[0][1] = 4;
    c[1][1] = 5;
    std::cout << "X1=" << c.to1DSpan() << "\n";
    ASSERT_EQ(c[0][0], 1);
    ASSERT_EQ(c[1][0], 2);
    ASSERT_EQ(c[0][1], 4);
    ASSERT_EQ(c[1][1], 5);
    c.resize(3, 2);
    std::cout << "X2=" << c.to1DSpan() << "\n";
    ASSERT_EQ(c[0][0], 1);
    ASSERT_EQ(c[1][0], 2);
    ASSERT_EQ(c[0][1], 4);
    ASSERT_EQ(c[1][1], 5);
    ASSERT_EQ(c[2][0], 0);
    ASSERT_EQ(c[2][1], 0);
    c[2][0] = 8;
    c[2][1] = 10;
    c.resize(6, 5);
    std::cout << "X3=" << c.to1DSpan() << "\n";
    ASSERT_EQ(c[0][0], 1);
    ASSERT_EQ(c[1][0], 2);
    ASSERT_EQ(c[0][1], 4);
    ASSERT_EQ(c[1][1], 5);
    ASSERT_EQ(c[2][0], 8);
    ASSERT_EQ(c[2][1], 10);
    for (int i = 0; i < 4; ++i) {
      ASSERT_EQ(c[i][2], 0);
      ASSERT_EQ(c[i][3], 0);
      ASSERT_EQ(c[i][4], 0);
    }
    for (int j = 0; j < 5; ++j) {
      ASSERT_EQ(c[3][j], 0);
      ASSERT_EQ(c[4][j], 0);
      ASSERT_EQ(c[5][j], 0);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
// Instancie explicitement les classes tableaux pour garantir
// que toutes les méthodes fonctionnent
template class UniqueArray2<IntSubClass>;
template class SharedArray2<IntSubClass>;
template class Array2<IntSubClass>;
}
