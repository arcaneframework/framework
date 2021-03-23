// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* mathlink.h                                                       (C) 2013 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef MATHEMATICA_LINK_H
#define MATHEMATICA_LINK_H

ARCANE_BEGIN_NAMESPACE

class mathlink : public AbstractService{
public:
  mathlink(const ServiceBuildInfo &);
  ~mathlink();
public:  
  void link();
  void unlink();
public:
  Integer Prime(Integer);
private:
  void skipAnyPacketsBeforeTheFirstReturnPacket();
  void error();
private:
  void tests();
  void testFactorInteger(Int64);
  void testLinearProgramming(ArrayView<Integer>);
private:
  ISubDomain *m_sub_domain;
  MLENV mathenv;
  MLINK mathlnk;
  Timer *mathtmr;
};
ARCANE_END_NAMESPACE
#endif // MATHEMATICA_LINK_H
