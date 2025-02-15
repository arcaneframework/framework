// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/ReferenceCounter.h"
#include "arccore/base/Ref.h"
#include "arccore/base/ReferenceCounterImpl.h"

#include <iostream>

using namespace Arccore;

namespace
{
struct StatInfo
{
  bool is_destroyed = false;
  int nb_add = 0;
  int nb_remove = 0;
  // Vérifie que 'is_destroyed' est vrai et qu'on a fait
  // le bon nombre d'appel à 'nb_add' et 'nb_remove'.
  bool checkValid(int nb_call)
  {
    if (nb_add!=nb_call){
      std::cout << "Bad nb_add n=" << nb_add << " expected=" << nb_call << "\n";
      return false;
    }
    if (nb_remove!=nb_call){
      std::cout << "Bad nb_remove n=" << nb_remove << " expected=" << nb_call << "\n";
      return false;
    }
    return is_destroyed;
  }
};
}

struct tag_ref_value {};

class Simple1
{
 public:
  typedef ReferenceCounterTag ReferenceCounterTagType;
 public:
  Simple1(StatInfo* stat_info) : m_nb_ref(0), m_stat_info(stat_info){}
  ~Simple1(){ m_stat_info->is_destroyed = true; }
 public:
  void addReference()
  {
    ++m_nb_ref;
    ++m_stat_info->nb_add;
    std::cout << "ADD REFERENCE r=" << m_nb_ref << "\n";
  }
  void removeReference()
  {
    --m_nb_ref;
    ++m_stat_info->nb_remove;
    std::cout << "REMOVE REFERENCE r=" << m_nb_ref << "\n";
    if (m_nb_ref==0){
      std::cout << "DESTROY!\n";
      delete this;
    }
  }
  Int32 m_nb_ref;
  StatInfo* m_stat_info;
};
// Test sans compteur de référence (avec std::shared_ptr).
class Simple2
{
 public:
  Simple2(StatInfo* stat_info) : m_stat_info(stat_info){}
  ~Simple2(){ m_stat_info->is_destroyed = true; }
 private:
  StatInfo* m_stat_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename RefType> void
_doTest1(const RefType& ref_type)
{
  {
    RefType s3;
    {
      RefType s1(ref_type);
      RefType s2 = s1;
      {
        s3 = s2;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Teste si le compteur de référence détruit bien l'instance.
TEST(ReferenceCounter, Misc)
{
  typedef ReferenceCounter<Simple1> Simple1Reference;
  {
    StatInfo stat_info;
    _doTest1(Simple1Reference(new Simple1(&stat_info)));
    ASSERT_TRUE(stat_info.checkValid(4)) << "Bad destroy1";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Teste si le compteur de référence détruit bien l'instance.
TEST(ReferenceCounter, Ref)
{
  {
    StatInfo stat_info;
    _doTest1(makeRef(new Simple1(&stat_info)));
    ASSERT_TRUE(stat_info.checkValid(4)) << "Bad destroy2";
  }
  {
    StatInfo stat_info;
    _doTest1(createRef<Simple2>(&stat_info));
    ASSERT_TRUE(stat_info.checkValid(0)) << "Bad destroy3";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Test1
{
class ITestClassWithDeleter
{
 public:

  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();
  virtual ~ITestClassWithDeleter() = default;

 public:

  virtual bool func1() = 0;
  virtual void func2() = 0;
};

class TestClassWithDeleter
: public ReferenceCounterImpl
, public ITestClassWithDeleter
{
 public:

  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  bool func1() override { return true; }
  void func2() override {}

 public:

  Int32 m_padding4 = 0;
  Int64 m_padding3 = 0;
};
} // namespace Test1

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Teste si le compteur de référence détruit bien l'instance.
TEST(ReferenceCounter, RefWithDeleter)
{
  try{
    using namespace Test1;
    Ref<ITestClassWithDeleter> myx2;
    {
      Ref<ITestClassWithDeleter> myx1 = makeRef<TestClassWithDeleter>(new TestClassWithDeleter());
      myx2 = myx1;
    }
    myx2.reset();
    Arcane::Internal::ExternalRef external_ref;
    {
      auto* ptr1 = new TestClassWithDeleter();
      Ref<ITestClassWithDeleter> x4 = Ref<ITestClassWithDeleter>::createWithHandle(ptr1,external_ref);
    }
    {
      Ref<ITestClassWithDeleter> myx3 = createRef<TestClassWithDeleter>();
      myx2 = myx3;
      bool is_ok = false;
      if (myx3)
        is_ok = true;
      ASSERT_TRUE(is_ok);
      ASSERT_FALSE(!myx3);
    }
    {
      Ref<ITestClassWithDeleter> myx4;
      bool is_null = true;
      if (myx4)
        is_null = false;
      ASSERT_TRUE(is_null);
      ASSERT_TRUE(!myx4);
    }
  }
  catch(const std::exception& ex){
    std::cerr << "Exception ex=" << ex.what() << "\n";
    throw;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
