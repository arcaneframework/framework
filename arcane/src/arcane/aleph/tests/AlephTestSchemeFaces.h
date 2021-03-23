// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephTestSchemeFaces.h                                           (C) 2011 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ALEPH_TEST_SCHEME_FACES_H
#define ALEPH_TEST_SCHEME_FACES_H

#include "arcane/aleph/tests/AlephTest.h"
#include "arcane/aleph/tests/AlephTestSchemeFaces_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

class CaseOptionsAlephTestModule;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

class AlephTestSchemeFaces : public ArcaneAlephTestSchemeFacesObject
{
 public:
  AlephTestSchemeFaces(const ServiceBuildInfo&);
  ~AlephTestSchemeFaces(void);

 public:
  virtual void boundaries(ArcaneTest::CaseOptionsAlephTestModule*);
  virtual void preFetchNumElementsForEachRow(IntegerArray&, const Integer);
  virtual void setValues(const Real, AlephMatrix*);
  virtual bool amrRefine(RealArray&, const Real);
  virtual bool amrCoarsen(RealArray&, const Real);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ALEPH_TEST_SCHEME_FACES_H
