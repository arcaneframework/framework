// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IServiceInterface.h                                         (C) 2000-2023 */
/*                                                                           */
/* Interfaces pour les tests des services.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_TEST_ISERVICEINTERFACE_H
#define ARCANE_TEST_ISERVICEINTERFACE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IServiceInterface1
{
 public:

  virtual ~IServiceInterface1() = default;

 public:

  virtual Arccore::Integer value() =0;
  virtual void* getPointer1() =0;
  virtual Arccore::String implementationName() const =0;
  virtual Arccore::String meshName() const =0;
  virtual void checkSubMesh(const Arccore::String&);
  virtual void checkDynamicCreation();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IServiceInterface2
{
 public:
  virtual ~IServiceInterface2(){}
  virtual void* getPointer2() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IServiceInterface3
{
 public:
  virtual ~IServiceInterface3(){}
  virtual void* getPointer3() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IServiceInterface4
{
 public:
  virtual ~IServiceInterface4(){}
  virtual void* getPointer4() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IServiceInterface5
{
 public:
  virtual ~IServiceInterface5(){}
  virtual void* getPointer5() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IServiceInterface6
{
 public:
  virtual ~IServiceInterface6(){}
  virtual void* getPointer6() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

