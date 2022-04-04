// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TestUnitTest.cc                                             (C) 2000-2020 */
/*                                                                           */
/* Service de test pour les tests unitaires.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ServiceBuildInfo.h"
#include "arcane/tests/ArcaneTestGlobal.h"

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du service
 */
class ITestBidonInterface
{
 public:

  virtual ~ITestBidonInterface() = default;

 public:

  virtual Real compute() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/TestUnitTest_axl.h"
#include "arcane/utils/TraceInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test des tests unitaires
 */
class TestUnitTest
: public ITestBidonInterface
, public ArcaneTestUnitTestObject
{
 public:

  explicit TestUnitTest(const ServiceBuildInfo& cb);
  ~TestUnitTest();

 public:

  // le boulot normal du service
  Real compute() override;

 public:

  // les méthodes de test
  void setUpForClass() override;
  void tearDownForClass() override;
  void setUp() override;
  void tearDown() override;
  void myTestMethod1() override;
  void myTestMethod2() override;
  void myTestMethod3() override;
  void myTestMethod4() override;
  void myTestMethod5() override;
  void myTestMethod1Parallel() override;
  void myTestMethod2Parallel() override;
  void myTestMethod3Parallel() override;
  void myTestMethod4Parallel() override;
  void myTestMethod5Parallel() override;

 private:

  Real m_my_small_double;
  Int32 m_my_small_integer;
  IParallelMng* m_parallel_mng = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TestUnitTest::
TestUnitTest(const ServiceBuildInfo& sbi)
: ArcaneTestUnitTestObject(sbi)
, m_my_small_double(1.0e-15)
, m_my_small_integer(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TestUnitTest::
~TestUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TestUnitTest::
setUpForClass()
{
	info() << "setUpForClass ";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TestUnitTest::
tearDownForClass()
{
	info() << "tearDownForClass ";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TestUnitTest::
setUp()
{
	info() << "setUp ";
  m_parallel_mng = subDomain()->parallelMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TestUnitTest::
tearDown()
{
	info() << "tearDown ";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// En utilisant les macros...
void TestUnitTest::
myTestMethod1()
{
	ASSERT_EQUAL(5, options()->myInt());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// En utilisant les macros...
void TestUnitTest::
myTestMethod1Parallel()
{
	PARALLEL_ASSERT_EQUAL(5, options()->myInt(), m_parallel_mng);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// En n'utilisant pas les macros...
void TestUnitTest::
myTestMethod2()
{
	ASSERT_EQUAL(5.5, options()->myDouble());
	ASSERT_NEARLY_EQUAL(5.5, options()->myDouble());
	ASSERT_NEARLY_EQUAL_EPSILON(5.5, options()->myDouble(),1e-10);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// En n'utilisant pas les macros...
void TestUnitTest::
myTestMethod2Parallel()
{
	PARALLEL_ASSERT_EQUAL(5.5, options()->myDouble(), m_parallel_mng);
	PARALLEL_ASSERT_NEARLY_EQUAL(5.5, options()->myDouble(), m_parallel_mng);
	PARALLEL_ASSERT_NEARLY_EQUAL_EPSILON(5.5, options()->myDouble(),1e-10, m_parallel_mng);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TestUnitTest::
myTestMethod3()
{
	ASSERT_TRUE(options()->myBoolean());
	ASSERT_EQUAL(true, options()->myBoolean());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TestUnitTest::
myTestMethod3Parallel()
{
	PARALLEL_ASSERT_TRUE(options()->myBoolean(), m_parallel_mng);
	PARALLEL_ASSERT_EQUAL(true, options()->myBoolean(), m_parallel_mng);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TestUnitTest::
myTestMethod4()
{
	ASSERT_NEARLY_ZERO(m_my_small_double);
	ASSERT_NEARLY_ZERO(m_my_small_integer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TestUnitTest::
myTestMethod4Parallel()
{
	PARALLEL_ASSERT_NEARLY_ZERO(m_my_small_double, m_parallel_mng);
	PARALLEL_ASSERT_NEARLY_ZERO(m_my_small_integer, m_parallel_mng);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TestUnitTest::
myTestMethod5()
{
	ASSERT_NEARLY_ZERO_EPSILON(m_my_small_double*5000.0,1.0e-11);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TestUnitTest::
myTestMethod5Parallel()
{
	PARALLEL_ASSERT_NEARLY_ZERO_EPSILON(m_my_small_double*5000.0,1.0e-11, m_parallel_mng);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real TestUnitTest::
compute()
{
	info() << "compute";
	return options()->myDouble();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_TESTUNITTEST(TestUnitTest,TestUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
