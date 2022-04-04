// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandardCaseFunction.h                                      (C) 2000-2011 */
/*                                                                           */
/* Classe gérant une fonction standard du jeu de données.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STANDARDCASEFUNCTION_H
#define ARCANE_STANDARDCASEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/CaseFunction.h"
#include "arcane/IStandardFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe gérant une fonction standard du jeu de données.
 *
 * \ingroup CaseOption
 *
 * Cette classe doit être héritée et la classe dérivée doit surcharger
 * l'une des méthodes qui renvoit un IFunctionWithArgAndReturn2. Sans
 * surchage, toutes ces méthode retournent un pointeur nul. La classe
 * dérivée n'est pas obligée de surchager toutes les méthodes getFunctor*
 * mais peut se contenter de surcharger que ceux qu'elle souhaite.
 */
class ARCANE_CORE_EXPORT StandardCaseFunction
: public CaseFunction
, public IStandardFunction
{
 public:

  //! Construit une fonction du jeu de données.
  StandardCaseFunction(const CaseFunctionBuildInfo& info);
  virtual ~StandardCaseFunction();

 private:

  virtual void value(Real param,Real& v) const;
  virtual void value(Real param,Integer& v) const;
  virtual void value(Real param,bool& v) const;
  virtual void value(Real param,String& v) const;
  virtual void value(Real param,Real3& v) const;
  virtual void value(Integer param,Real& v) const;
  virtual void value(Integer param,Integer& v) const;
  virtual void value(Integer param,bool& v) const;
  virtual void value(Integer param,String& v) const;
  virtual void value(Integer param,Real3& v) const;

 public:
  
  virtual IBinaryMathFunctor<Real,Real,Real>* getFunctorRealRealToReal();
  virtual IBinaryMathFunctor<Real,Real3,Real>* getFunctorRealReal3ToReal();
  virtual IBinaryMathFunctor<Real,Real,Real3>* getFunctorRealRealToReal3();
  virtual IBinaryMathFunctor<Real,Real3,Real3>* getFunctorRealReal3ToReal3();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

