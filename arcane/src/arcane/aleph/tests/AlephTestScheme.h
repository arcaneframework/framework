// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephTestIScheme.h                                               (C) 2011 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ALEPH_TEST_SCHEME_H
#define ALEPH_TEST_SCHEME_H

ARCANETEST_BEGIN_NAMESPACE
using namespace Arcane;

class CaseOptionsAlephTestModule;

class AlephTestScheme
{
 public:
  AlephTestScheme(void){};
  virtual ~AlephTestScheme(void){};

 public:
  virtual void boundaries(ArcaneTest::CaseOptionsAlephTestModule*) = 0;
  virtual void preFetchNumElementsForEachRow(IntegerArray&, const Integer) = 0;
  virtual void setValues(const Real, AlephMatrix*) = 0;
  virtual bool amrRefine(RealArray&, const Real) = 0;
  virtual bool amrCoarsen(RealArray&, const Real) = 0;
};

ARCANETEST_END_NAMESPACE

#endif // ALEPH_TEST_SCHEME_H
