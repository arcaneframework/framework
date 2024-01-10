// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/Ref.h"
#include "arccore/base/ReferenceCounterImpl.h"

#include <string>

using namespace Arccore;

namespace MyTest
{

//! Classe de test utilisant le compteur de référence interne
class TestRefOwn;
class TestBaseType;
}

ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(MyTest::TestBaseType)

namespace MyTest
{
int global_nb_create = 0;
int global_nb_destroy = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TestBaseType
: public ReferenceCounterImpl
{
 public:
  typedef ReferenceCounterTag ReferenceCounterTagType;
 public:
  TestBaseType(int a,const std::string& b) : m_a(a), m_b(b)
  {
    std::cout << "CREATE ME this=" << this << "\n";
    ++global_nb_create;
  }
  TestBaseType(const TestBaseType& x) : m_a(x.m_a), m_b(x.m_b)
  {
    std::cout << "CREATE ME (copy) this=" << this << "\n";
  }
  virtual ~TestBaseType()
  {
    std::cout << "DELETE ME this=" << this << "\n";
    ++global_nb_destroy;
  }
 public:
  int pa() const { return m_a; }
  const std::string& pb() const { return m_b; }
  void print(const std::string& x) const
  {
    std::cout << x << " A=" << pa() << " B=" << pb()
              << " this=" << this << "\n";
  }
 private:
  int m_a;
  std::string m_b;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TestBaseTypeNoRef
{
 public:
  TestBaseTypeNoRef(int a,const std::string& b) : m_a(a), m_b(b)
  {
    std::cout << "CREATE ME this=" << this << "\n";
    ++global_nb_create;
  }
  virtual ~TestBaseTypeNoRef()
  {
    std::cout << "DELETE ME this=" << this << "\n";
    ++global_nb_destroy;
  }
 public:
  int pa() const { return m_a; }
  const std::string& pb() const { return m_b; }
  void print(const std::string& x) const
  {
    std::cout << x << " A=" << pa() << " B=" << pb()
              << " this=" << this << "\n";
  }
 private:
  int m_a;
  std::string m_b;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TestRefOwn
: public TestBaseType
{
 public:
  typedef TestBaseType BaseType;
 public:
  TestRefOwn(int a,const std::string& b) : TestBaseType(a,b){}
  ~TestRefOwn() override
  {
    std::cout << "DELETE ME (TestRefOwn) this=" << this << "\n";
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Classe de test utilisant un shared_ptr
class TestRefSharedPtr
: public TestBaseTypeNoRef
{
 public:
  typedef TestBaseTypeNoRef BaseType;
 public:
  TestRefSharedPtr(int a,const std::string& b) : TestBaseTypeNoRef(a,b){}
};

class TestRefMacroInternal
: public ReferenceCounterImpl
{
  ARCCORE_INTERNAL_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();
};

}

using namespace MyTest;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ClassType,int id> void
_doTest1Helper()
{
  std::cout << "** ** BEGIN_TEST\n";

  typedef Ref<ClassType> RefType;
  typedef typename ClassType::BaseType BaseType;
  int a = 3;
  std::string b = "XYZ";

  {
    RefType ref0(makeRef(new ClassType(a,b)));
    ref0->print("X0");
  }

  RefType ref1(RefType::template createRef<ClassType>(a,b));
  ref1->print("X1");
  {
    RefType ref2(ref1);
    ref2->print("X2");
    Ref<BaseType> ref3(ref1);
    ref3->print("X3");
  }
  if constexpr (id == 1){
    ClassType* ct = ref1.get();
    auto z = makeRefFromInstance<TestBaseType>(ct);
  }
  {
    RefType null_ref_type;
    ASSERT_EQ(null_ref_type.get(),nullptr);
    if constexpr (id==0){
      ClassType* ct2 = null_ref_type._release();
      ASSERT_EQ(ct2,nullptr);
    }
  }
  {
    ClassType* ct = nullptr;
    RefType null_ref_type(makeRef(ct));
    ASSERT_EQ(null_ref_type.get(),nullptr);
    if constexpr (id==0){
      ClassType* ct2 = null_ref_type._release();
      ASSERT_EQ(ct2,nullptr);
    }
  }
  std::cout << "DoTestRelease\n";
  {
    RefType ref_ct(makeRef(new ClassType(a,b)));
    ASSERT_NE(ref_ct.get(),nullptr);
    ASSERT_EQ(ref_ct->pa(),a);
    ASSERT_EQ(ref_ct->pb(),b);
    if constexpr (id==0){
      ClassType* ct2 = ref_ct._release();
      ASSERT_NE(ct2,nullptr);
      ASSERT_EQ(ct2->pa(),a);
      ASSERT_EQ(ct2->pb(),b);
      delete ct2;
    }
  }
  std::cout << "** ** END_TEST\n";
}

template<typename ClassType,int id> void
_doTest1()
{
  global_nb_create = global_nb_destroy = 0;
  _doTest1Helper<ClassType,id>();
  ASSERT_EQ(global_nb_create,3);
  ASSERT_EQ(global_nb_create,global_nb_destroy);
}

void
_doTest2()
{
  TestRefOwn* x1 = new TestRefOwn(1,"ZXB");
  delete x1;

  {
    auto x2 = makeRef<TestRefMacroInternal>(new TestRefMacroInternal());
  }
}

namespace Arccore
{
ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(MyTest::TestBaseType);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Teste si le compteur de référence détruit bien l'instance.
TEST(Ref, Misc)
{
  _doTest1<TestRefOwn,1>();
  _doTest1<TestRefSharedPtr,0>();
  _doTest2();
}
