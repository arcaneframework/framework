// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VoronoiTest.cc                                              (C) 2000-2009 */
/*                                                                           */
/* Service du test des maillages Voronoï.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/BasicUnitTest.h"

#include "arcane/IParallelMng.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IVariableMng.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/VoronoiTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test du maillage Voronoï
 */
class VoronoiTest
: public ArcaneVoronoiTestObject
{
public:

  VoronoiTest(const ServiceBuildInfo& cb);
  ~VoronoiTest();

 public:

  virtual void initializeTest();
  virtual void executeTest();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_VORONOITEST(VoronoiTest,VoronoiTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VoronoiTest::
VoronoiTest(const ServiceBuildInfo& mb)
: ArcaneVoronoiTestObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VoronoiTest::
~VoronoiTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VoronoiTest::
executeTest()
{
  ENUMERATE_CELL(icell,allCells())
    m_cell_flags[icell] = Real(icell->uniqueId().asInt64());

  ENUMERATE_CELL(icell,ownCells())
    m_domain_id[icell] = subDomain()->subDomainId();

  ENUMERATE_FACE(iface,allFaces())
    m_face_flags[iface] = Real(iface->uniqueId().asInt64());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VoronoiTest::
initializeTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
