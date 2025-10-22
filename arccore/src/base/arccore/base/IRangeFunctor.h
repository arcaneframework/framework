// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRangeFunctor.h                                             (C) 2000-2025 */
/*                                                                           */
/* Interface d'un fonctor sur un interval d'itération.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_IRANGEFUNCTOR_H
#define ARCCORE_BASE_IRANGEFUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

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
class ARCCORE_BASE_EXPORT IRangeFunctor
{
 public:

  //! Libère les ressources
  virtual ~IRangeFunctor() = default;

 public:

  /*!
   * \brief Exécute la méthode associée.
   * \param begin indice du début de l'itération.
   * \param size nombre d'éléments à itérer.
   */
  virtual void executeFunctor(Int32 begin, Int32 size) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un fonctor sur un interval d'itération multi-dimensionnel
 * de dimension \a RankValue
 * \ingroup Core
 */
template <int RankValue>
class IMDRangeFunctor
{
 public:

  //! Libère les ressources
  virtual ~IMDRangeFunctor() = default;

 public:

  /*!
   * \brief Exécute la méthode associée.
   */
  virtual void executeFunctor(const ComplexForLoopRanges<RankValue>& loop_range) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
