// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandardCaseFunction.h                                      (C) 2000-2025 */
/*                                                                           */
/* Classe gérant une fonction standard du jeu de données.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_STANDARDCASEFUNCTION_H
#define ARCANE_CORE_STANDARDCASEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseFunction.h"
#include "arcane/core/IStandardFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe gérant une fonction standard du jeu de données.
 *
 * \ingroup CaseOption
 *
 * Cette classe doit être héritée et la classe dérivée doit surcharger
 * l'une des méthodes qui renvoie un IFunctionWithArgAndReturn2. Sans
 * surcharge, toutes ces méthode retournent un pointeur nul. La classe
 * dérivée n'est pas obligée de surcharger toutes les méthodes getFunctor*
 * mais peut se contenter de surcharger que ceux qu'elle souhaite.
 */
class ARCANE_CORE_EXPORT StandardCaseFunction
: public CaseFunction
, public IStandardFunction
{
 public:

  //! Construit une fonction du jeu de données.
  explicit StandardCaseFunction(const CaseFunctionBuildInfo& info);
  ~StandardCaseFunction() override;

 private:

  void value(Real param, Real& v) const override;
  void value(Real param, Integer& v) const override;
  void value(Real param, bool& v) const override;
  void value(Real param, String& v) const override;
  void value(Real param, Real3& v) const override;
  void value(Integer param, Real& v) const override;
  void value(Integer param, Integer& v) const override;
  void value(Integer param, bool& v) const override;
  void value(Integer param, String& v) const override;
  void value(Integer param, Real3& v) const override;

 public:

  // NOTE : Le mot clé 'virtual' est nécessaire pour SWIG
  // car cette classe a un double héritage et SWIG ne prend
  // que le premier et donc ne voit pas les méthodes suivantes
  // comme étant virtuelles.
  virtual IBinaryMathFunctor<Real, Real, Real>* getFunctorRealRealToReal() override;
  virtual IBinaryMathFunctor<Real, Real3, Real>* getFunctorRealReal3ToReal() override;
  virtual IBinaryMathFunctor<Real, Real, Real3>* getFunctorRealRealToReal3() override;
  virtual IBinaryMathFunctor<Real, Real3, Real3>* getFunctorRealReal3ToReal3() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
