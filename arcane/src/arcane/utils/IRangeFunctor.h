// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRangeFunctor.h                                             (C) 2000-2021 */
/*                                                                           */
/* Interface d'un fonctor sur un interval d'itération.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IRANGEFUNCTOR_H
#define ARCANE_UTILS_IRANGEFUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un fonctor sur un interval d'itération.
 * \ingroup Core
 */
class ARCANE_UTILS_EXPORT IRangeFunctor
{
 public:
	
  //! Libère les ressources
  virtual ~IRangeFunctor(){}

 public:

  /*!
   * \brief Exécute la méthode associée.
   * \param begin indice du début de l'itération.
   * \param size nombre d'éléments à itérer.
   */
  virtual void executeFunctor(Integer begin,Integer size) =0;  
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un fonctor sur un interval d'itération multi-dimensionnel
 * de dimension \a RankValue
 * \ingroup Core
 */
template<int RankValue>
class IMDRangeFunctor
{
 public:

  //! Libère les ressources
  virtual ~IMDRangeFunctor() = default;

 public:

  /*!
   * \brief Exécute la méthode associée.
   */
  virtual void executeFunctor(const ComplexForLoopRanges<RankValue>& loop_range) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
