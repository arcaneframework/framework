// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMathFunctor.h                                              (C) 2000-2011 */
/*                                                                           */
/* Interface d'un fonctor pour une fonction mathématiques.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IMATHFUNCTOR_H
#define ARCANE_UTILS_IMATHFUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une fonction mathématique binaire.
 */
template<typename Arg1,typename Arg2,typename ReturnType>
class IBinaryMathFunctor
{
 public:
	
  //! Libère les ressources
  virtual ~IBinaryMathFunctor(){}

 public:

  //! Exécute la méthode associé
  virtual ReturnType apply(Arg1 a1,Arg2 a2) =0;

  //! Exécute la méthode associé
  virtual void apply(ConstArrayView<Arg1> a1,ConstArrayView<Arg2> a2,ArrayView<ReturnType> result)
  {
    for( Integer i=0,n=result.size(); i<n; ++i )
      result[i] = apply(a1[i],a2[i]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

