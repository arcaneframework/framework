// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRangeFunctor.h                                             (C) 2000-2026 */
/*                                                                           */
/* Interface of a functor on an iteration interval.                          */
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
 * \brief Interface of a functor on an iteration interval.
 * \ingroup Core
 */
class ARCCORE_BASE_EXPORT IRangeFunctor
{
 public:

  //! Releases resources
  virtual ~IRangeFunctor() = default;

 public:

  /*!
   * \brief Executes the associated method.
   * \param begin index of the start of the iteration.
   * \param size number of elements to iterate.
   */
  virtual void executeFunctor(Int32 begin, Int32 size) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface of a functor on a multi-dimensional iteration interval
 * of dimension \a RankValue
 * \ingroup Core
 */
template <int RankValue>
class IMDRangeFunctor
{
 public:

  //! Releases resources
  virtual ~IMDRangeFunctor() = default;

 public:

  /*!
   * \brief Executes the associated method.
   */
  virtual void executeFunctor(const ComplexForLoopRanges<RankValue>& loop_range) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
