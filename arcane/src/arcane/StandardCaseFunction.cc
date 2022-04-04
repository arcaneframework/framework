// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandardCaseFunction.cc                                     (C) 2000-2016 */
/*                                                                           */
/* Classe gérant une fonction standard du jeu de données.                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/Real3.h"
#include "arcane/StandardCaseFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandardCaseFunction::
StandardCaseFunction(const CaseFunctionBuildInfo& info)
: CaseFunction(info)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandardCaseFunction::
~StandardCaseFunction()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardCaseFunction::
value(Real param,Real& v) const
{
  ARCANE_UNUSED(param);
  ARCANE_UNUSED(v);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardCaseFunction::
value(Real param,Integer& v) const
{
  ARCANE_UNUSED(param);
  ARCANE_UNUSED(v);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardCaseFunction::
value(Real param,bool& v) const
{
  ARCANE_UNUSED(param);
  ARCANE_UNUSED(v);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardCaseFunction::
value(Real param,String& v) const
{
  ARCANE_UNUSED(param);
  ARCANE_UNUSED(v);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardCaseFunction::
value(Real param,Real3& v) const
{
  ARCANE_UNUSED(param);
  ARCANE_UNUSED(v);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardCaseFunction::
value(Integer param,Real& v) const
{
  ARCANE_UNUSED(param);
  ARCANE_UNUSED(v);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardCaseFunction::
value(Integer param,Integer& v) const
{
  ARCANE_UNUSED(param);
  ARCANE_UNUSED(v);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardCaseFunction::
value(Integer param,bool& v) const
{
  ARCANE_UNUSED(param);
  ARCANE_UNUSED(v);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardCaseFunction::
value(Integer param,String& v) const
{
  ARCANE_UNUSED(param);
  ARCANE_UNUSED(v);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardCaseFunction::
value(Integer param,Real3& v) const
{
  ARCANE_UNUSED(param);
  ARCANE_UNUSED(v);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IBinaryMathFunctor<Real,Real,Real>* StandardCaseFunction::
getFunctorRealRealToReal()
{
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IBinaryMathFunctor<Real,Real3,Real>* StandardCaseFunction::
getFunctorRealReal3ToReal()
{
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IBinaryMathFunctor<Real,Real,Real3>* StandardCaseFunction::
getFunctorRealRealToReal3()
{
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IBinaryMathFunctor<Real,Real3,Real3>* StandardCaseFunction::
getFunctorRealReal3ToReal3()
{
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

