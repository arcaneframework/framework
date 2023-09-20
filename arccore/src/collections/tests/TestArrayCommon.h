// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef TEST_ARCCORE_COLLECTIONS_TESTARRAYCOMMON_H
#define TEST_ARCCORE_COLLECTIONS_TESTARRAYCOMMON_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <gtest/gtest.h>

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace TestArccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;
class IntSubClass
{
 public:

  IntSubClass(Integer v)
  : m_v(v)
  {}
  IntSubClass() = default;
  Integer m_v = 0;
  friend bool operator==(const IntSubClass& v, Integer iv) { return v.m_v == iv; }
  friend bool operator==(const IntSubClass& v1, const IntSubClass& v2) { return v1.m_v == v2.m_v; }
  friend bool operator!=(const IntSubClass& v1, const IntSubClass& v2) { return v1.m_v != v2.m_v; }
  friend std::ostream& operator<<(std::ostream& o, const IntSubClass& c)
  {
    o << c.m_v;
    return o;
  }
};
class IntSubClassNoPod
{
 public:

  IntSubClassNoPod() = default;
  explicit IntSubClassNoPod(Integer v)
  : m_v(v)
  {}
  //IntSubClassNoPod() : m_v(0) {}
  Integer m_v = 0;
  friend bool operator==(const IntSubClassNoPod& v, Integer iv) { return v.m_v == iv; }
  //friend bool operator==(const IntSubClassNoPod& v1,const IntSubClassNoPod& v2) { return v1.m_v==v2.m_v; }
  friend bool operator!=(const IntSubClassNoPod& v1, const IntSubClassNoPod& v2) { return v1.m_v != v2.m_v; }
  friend std::ostream& operator<<(std::ostream& o, IntSubClassNoPod c)
  {
    o << c.m_v;
    return o;
  }
};
} // namespace TestArccore
namespace Arccore
{
ARCCORE_DEFINE_ARRAY_PODTYPE(TestArccore::IntSubClass);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCCORE_UT_CHECK(expr,message) \
if ( ! (expr) )\
  throw Arccore::FatalErrorException((message))

namespace TestArccore
{
class IntPtrSubClass
{
 public:

  static Integer count;
  IntPtrSubClass(Integer v)
  : m_v(new Integer(v))
  {
    ++count;
  }
  IntPtrSubClass()
  : m_v(new Integer(0))
  {
    ++count;
  }
  ~IntPtrSubClass()
  {
    --count;
    delete m_v;
  }
  Integer* m_v;
  IntPtrSubClass(const IntPtrSubClass& v)
  : m_v(new Integer(*v.m_v))
  {
    ++count;
  }
  void operator=(const IntPtrSubClass& v)
  {
    Integer* n = new Integer(*v.m_v);
    delete m_v;
    m_v = n;
  }
  bool operator==(Integer iv) const
  {
    //cout << "** COMPARE: " << *m_v << " v=" << iv << '\n';
    return *m_v == iv;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> inline
void _checkSameInfoArray2(const Array2<T>& a, const Array2<T>& b)
{
  ASSERT_EQ(a.allocator(), b.allocator());
  ASSERT_EQ(a.totalNbElement(), b.totalNbElement());
  ASSERT_EQ(a.dim1Size(), b.dim1Size());
  ASSERT_EQ(a.dim2Size(), b.dim2Size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace TestArccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
