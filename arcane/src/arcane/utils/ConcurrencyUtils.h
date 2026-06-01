// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConcurrencyUtils.h                                          (C) 2000-2025 */
/*                                                                           */
/* Classes managing concurrency (tasks, parallel loops, ...)                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_CONCURRENCYUTILS_H
#define ARCANE_UTILS_CONCURRENCYUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arccore/concurrency/ParallelFor.h"
#include "arccore/concurrency/TaskFactory.h"
#include "arccore/concurrency/ITaskImplementation.h"
#include "arccore/concurrency/Task.h"
#include "arccore/base/ForLoopRunInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the lambda function \a lambda_function
 * concurrently over the iteration range given by \a loop_ranges.
 */
template<int RankValue,typename LambdaType,typename... ReducerArgs> inline void
arcaneParallelFor(const ComplexForLoopRanges<RankValue>& loop_ranges,
                  const ForLoopRunInfo& run_info,
                  const LambdaType& lambda_function,
                  const ReducerArgs&... reducer_args)
{
  arccoreParallelFor(loop_ranges,run_info,lambda_function,reducer_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the lambda function \a lambda_function
 * concurrently over the iteration range given by \a loop_ranges.
 */
template<int RankValue,typename LambdaType,typename... ReducerArgs> inline void
arcaneParallelFor(const ComplexForLoopRanges<RankValue>& loop_ranges,
                  const ParallelLoopOptions& options,
                  const LambdaType& lambda_function,
                  const ReducerArgs&... reducer_args)
{
  arccoreParallelFor(loop_ranges,ForLoopRunInfo(options),lambda_function,reducer_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the lambda function \a lambda_function
 * concurrently over the iteration range given by \a loop_ranges.
 */
template <int RankValue, typename LambdaType, typename... ReducerArgs> inline void
arcaneParallelFor(const SimpleForLoopRanges<RankValue>& loop_ranges,
                  const ForLoopRunInfo& run_info,
                  const LambdaType& lambda_function,
                  const ReducerArgs&... reducer_args)
{
  arccoreParallelFor(loop_ranges,run_info,lambda_function,reducer_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the lambda function \a lambda_function
 * concurrently over the iteration range given by \a loop_ranges.
 */
template <int RankValue, typename LambdaType, typename... ReducerArgs> inline void
arcaneParallelFor(const SimpleForLoopRanges<RankValue>& loop_ranges,
                  const ParallelLoopOptions& options,
                  const LambdaType& lambda_function,
                  const ReducerArgs&... reducer_args)
{
  arccoreParallelFor(loop_ranges,options,lambda_function,reducer_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the lambda function \a lambda_function
 * concurrently over the iteration range given by \a loop_ranges.
 */
template<int RankValue,typename LambdaType> inline void
arcaneParallelFor(const ComplexForLoopRanges<RankValue>& loop_ranges,
                  const LambdaType& lambda_function)
{
  arccoreParallelFor(loop_ranges,lambda_function);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the lambda function \a lambda_function
 * concurrently over the iteration range given by \a loop_ranges.
 */
template<int RankValue,typename LambdaType> inline void
arcaneParallelFor(const SimpleForLoopRanges<RankValue>& loop_ranges,
                  const LambdaType& lambda_function)
{
  arccoreParallelFor(loop_ranges,lambda_function);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
