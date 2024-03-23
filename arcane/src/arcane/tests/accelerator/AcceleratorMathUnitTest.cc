// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorMathUnitTest.cc                                  (C) 2000-2024 */
/*                                                                           */
/* Service de test des fonctions mathématiques pour les accélérateurs.       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ValueChecker.h"

#include "arcane/utils/NumArray.h"
#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ServiceFactory.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/NumArrayViews.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;
namespace ax = Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de la classe 'AcceleratorViews'.
 */
class AcceleratorMathUnitTest
: public BasicUnitTest
{
 public:

  explicit AcceleratorMathUnitTest(const ServiceBuildInfo& cb);

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  ax::Runner m_runner;

 public:

  void _executeTest1();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(AcceleratorMathUnitTest, IUnitTest,
                                           AcceleratorMathUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorMathUnitTest::
AcceleratorMathUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorMathUnitTest::
initializeTest()
{
  m_runner = *(subDomain()->acceleratorMng()->defaultRunner());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorMathUnitTest::
executeTest()
{
  _executeTest1();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorMathUnitTest::
_executeTest1()
{
  auto queue = makeQueue(m_runner);

  const Int32 nb_test = 2;

  NumArray<bool, MDDim1> test_results(nb_test);

  Real3x3 test1_result(Real3(17.0, 11.0, 20.0),
                       Real3(26.0, 16.0, 38.0),
                       Real3(12.0, 14.0, 14.0));

  Real3x3 test2_result(Real3(17.0, 26.0, 12.0),
                       Real3(11.0, 16.0, 14.0),
                       Real3(20.0, 38.0, 14.0));
  {
    auto command = makeCommand(queue);
    auto test_results_view = viewOut(command, test_results);
    command << RUNCOMMAND_LOOP1(iter, nb_test)
    {
      auto [i] = iter();
      Real3 a(1.0, 2.0, 3.0);
      Real3 b(4.0, 1.0, 5.0);
      Real3 c(2.0, 3.0, 1.0);
      Real3x3 mat1(a, b, c);
      Real3x3 mat2(a, c, b);
      if (i == 0) {
        Real3x3 product = math::matrixProduct(mat1, mat2);
        //std::cout << "RESULT=" << product << "\n";
        test_results_view[i] = (product == test1_result);
      }
      if (i == 1) {
        Real3x3 product = math::matrixProduct(mat1, mat2);
        Real3x3 t = math::matrixTranspose(product);
        test_results_view[i] = (t == test2_result);
      }
    };
    bool has_bad = false;
    for (Int32 i = 0; i < nb_test; ++i) {
      info() << "Result I=" << i << " v=" << test_results[i];
      if (!test_results[i])
        has_bad = true;
    }
    if (has_bad)
      ARCANE_FATAL("Bad tests");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
