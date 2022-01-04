// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MKLVectorMathService.cc                                     (C) 2000-2010 */
/*                                                                           */
/* Operations mathematiques sur des vecteurs via la MKL.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Array.h"

#include "arcane/FactoryService.h"
#include "arcane/AbstractService.h"

#include <mkl.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MKLVectorMathService
: public AbstractService
{
 public:

  MKLVectorMathService(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {
  }

  //<! Libère les ressources
  virtual ~MKLVectorMathService()
  {
  }

 public:

  virtual void build()
  {
    printVersion();
  }

  void printVersion()
  {
    MKLVersion v;
    MKL_Get_Version(&v); /* Returns information about the active version of the Intel MKL software */

    info() << "Intel MKL version=" << v.MajorVersion
           << "." << v.MinorVersion
           << "." << v.UpdateVersion;
  }

  void test1()
  {
    Integer n = 1000;
    UniqueArray<double> a1(n);
    UniqueArray<double> a2(n);
    UniqueArray<double> c(n);
    for( Integer i=0; i<n; ++i ){
      a1[i] = 1.0 * (double)i;
      a2[i] = 3.0 * (double)i;
    }
    MKL_INT vn = (MKL_INT)n;
    vdAdd(vn,a1.data(),a2.data(),c.data());
  }

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
