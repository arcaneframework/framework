// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/AutoRef.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

class MyClass
{
 public:

  explicit MyClass(Int32 x)
  : m_value(x)
  {
    std::cout << "CREATE_REF value=" << m_value << "\n";
  }
  ~MyClass()
  {
    std::cout << "DESTROY_REF value=" << m_value << "\n";
  }

  void addRef()
  {
    ++m_nb_ref;
    std::cout << "ADD_REF n=" << m_nb_ref << " value=" << m_value << "\n";
  }

  void removeRef()
  {
    --m_nb_ref;
    std::cout << "REMOVE_REF n=" << m_nb_ref << " value=" << m_value << "\n";
    if (m_nb_ref == 0) {
      delete this;
    }
  }

 private:

  Int32 m_value = 0;
  Int32 m_nb_ref = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AutoRef2<MyClass> _buildOne(Int32 v)
{
  return AutoRef2<MyClass>(new MyClass(v));
}

AutoRef2<MyClass> _buildTwo(Int32 v1, Int32 v2)
{
  AutoRef2<MyClass> ref_v1(new MyClass(v1));
  AutoRef2<MyClass> ref_v2(new MyClass(v2));
  ref_v1 = ref_v2;
  return ref_v1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestAutoRef, Misc)
{
  AutoRef2<MyClass> ref1(new MyClass(25));
  AutoRef2<MyClass> ref2(new MyClass(12));
  ref1 = _buildOne(46);
  AutoRef2<MyClass> ref3(_buildOne(17));
  AutoRef2<MyClass> ref4(std::move(_buildTwo(57, 16)));
  AutoRef2<MyClass> ref5(ref4);
  std::cout << "END_OF_TEST\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
