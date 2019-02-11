// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include <gtest/gtest.h>

#include "arccore/base/ReferenceCounter.h"

#include <iostream>

using namespace Arccore;

class Simple1
{
 public:
  Simple1(bool* is_destroyed) : m_nb_ref(0), m_is_destroyed(is_destroyed){}
 public:
  void addReference()
  {
    ++m_nb_ref;
    std::cout << "ADD REFERENCE r=" << m_nb_ref << "\n";
  }
  void removeReference()
  {
    --m_nb_ref;
    std::cout << "REMOVE REFERENCE r=" << m_nb_ref << "\n";
    if (m_nb_ref==0){
      (*m_is_destroyed) = true;
      std::cout << "DESTROY!\n";
      delete this;
    }
  }
  Int32 m_nb_ref;
  bool* m_is_destroyed;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Teste si le compteur de référence détruit bien l'instance.
TEST(ReferenceCounter, Misc)
{
  typedef ReferenceCounter<Simple1> RefSimple1;
  bool is_destroyed(false);
  {
    RefSimple1 s3;
    {
      RefSimple1 s1(new Simple1(&is_destroyed));
      RefSimple1 s2 = s1;
      {
      s3 = s2;
      }
    }
  }
  ASSERT_TRUE(is_destroyed) << "Bad destroy";
}
