// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshBuilderCustomiserUnitTest.cc                            (C) 2000-2026 */
/*                                                                           */
/* Test of interface 'IMeshBuilderCustomiser'.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/FactoryService.h"

#include "arcane/core/ItemTypeMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IMeshMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Entity type test service
 */
class MeshBuilderCustomiserUnitTest
: public BasicUnitTest
{
 public:

  explicit MeshBuilderCustomiserUnitTest(const ServiceBuildInfo& cb);

 public:

  void initializeTest() override {}
  void executeTest() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(MeshBuilderCustomiserUnitTest,
                                           IUnitTest, MeshBuilderCustomiserUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshBuilderCustomiserUnitTest::
MeshBuilderCustomiserUnitTest(const ServiceBuildInfo& mb)
: BasicUnitTest(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshBuilderCustomiserUnitTest::
executeTest()
{
  IParallelMng* orig_pm = subDomain()->parallelMng();
  IMeshMng* mm = subDomain()->meshMng();
  IMesh* mesh0 = mm->findMeshHandle("Mesh0").mesh();
  IParallelMng* mesh0_pm = mesh0->parallelMng();

  if (mesh0_pm != orig_pm)
    ARCANE_FATAL("Not same IParallelMng for Mesh0 and subDomain()");

  // Not all ranks have Mesh1
  MeshHandle* mesh1_handle = mm->findMeshHandle("Mesh1", false);
  if (mesh1_handle) {
    IParallelMng* mesh1_pm = mesh1_handle->mesh()->parallelMng();
    // In the test case we have 8 PE in the original IParallelMng and a divider of 4.
    if (orig_pm->isParallel() && mesh1_pm->commSize() != 2)
      ARCANE_FATAL("Bad size for new IParallelMng");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
