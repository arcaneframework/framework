// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelFor.h                                               (C) 2000-2025 */
/*                                                                           */
/* Parallel loop management.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_PARALLELFOR_H
#define ARCCORE_BASE_PARALLELFOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/TaskFactory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Characteristics of a multi-thread 1D loop.
 *
 * This class allows specifying the options of a loop to be parallelized
 * in multi-thread mode.
 */
class ARCCORE_CONCURRENCY_EXPORT ParallelFor1DLoopInfo
{
 public:

  using ThatClass = ParallelFor1DLoopInfo;

 public:

  ParallelFor1DLoopInfo(Int32 begin, Int32 size, IRangeFunctor* functor)
  : m_begin(begin)
  , m_size(size)
  , m_functor(functor)
  {}
  ParallelFor1DLoopInfo(Int32 begin, Int32 size, IRangeFunctor* functor, const ForLoopRunInfo& run_info)
  : m_run_info(run_info)
  , m_begin(begin)
  , m_size(size)
  , m_functor(functor)
  {}
  ParallelFor1DLoopInfo(Int32 begin, Int32 size, Int32 block_size, IRangeFunctor* functor)
  : m_begin(begin)
  , m_size(size)
  , m_functor(functor)
  {
    ParallelLoopOptions opts(TaskFactory::defaultParallelLoopOptions());
    opts.setGrainSize(block_size);
    m_run_info.addOptions(opts);
  }

 public:

  Int32 beginIndex() const { return m_begin; }
  Int32 size() const { return m_size; }
  IRangeFunctor* functor() const { return m_functor; }
  ForLoopRunInfo& runInfo() { return m_run_info; }
  const ForLoopRunInfo& runInfo() const { return m_run_info; }

 private:

  ForLoopRunInfo m_run_info;
  Int32 m_begin = 0;
  Int32 m_size = 0;
  IRangeFunctor* m_functor = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the lambda function \a lambda_function concurrently
 * over the iteration interval given by \a loop_ranges.
 */
template <int RankValue, typename LambdaType, typename... ReducerArgs> inline void
arccoreParallelFor(const ComplexForLoopRanges<RankValue>& loop_ranges,
                   const ForLoopRunInfo& run_info,
                   const LambdaType& lambda_function,
                   const ReducerArgs&... reducer_args)
{
  // Modified Arcane 3.7.9 (September 2022)
  // Performs a copy to privatize the lambda values to the current thread.
  // This is necessary so that objects like reducers are properly taken
  // into account.
  // TODO: check if we could perform the copy only once per thread
  // if this copy becomes costly.
  // NOTE: Starting from version 3.12.15 (April 2024), with the new version
  // of reducers (Reduce2), this privatization is no longer useful. Once
  // we have removed the old classes managing reductions (Reduce),
  // we can remove this privatization
  auto xfunc = [&lambda_function, reducer_args...](const ComplexForLoopRanges<RankValue>& sub_bounds) {
    using Type = typename std::remove_reference<LambdaType>::type;
    Type private_lambda(lambda_function);
    arccoreSequentialFor(sub_bounds, private_lambda, reducer_args...);
  };
  LambdaMDRangeFunctor<RankValue, decltype(xfunc)> ipf(xfunc);
  TaskFactory::executeParallelFor(loop_ranges, run_info, &ipf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the lambda function \a lambda_function concurrently
 * over the iteration interval given by \a loop_ranges.
 */
template <int RankValue, typename LambdaType, typename... ReducerArgs> inline void
arccoreParallelFor(const ComplexForLoopRanges<RankValue>& loop_ranges,
                   const ParallelLoopOptions& options,
                   const LambdaType& lambda_function,
                   const ReducerArgs&... reducer_args)
{
  arccoreParallelFor(loop_ranges, ForLoopRunInfo(options), lambda_function, reducer_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the lambda function \a lambda_function concurrently
 * over the iteration interval given by \a loop_ranges.
 */
template <int RankValue, typename LambdaType, typename... ReducerArgs> inline void
arccoreParallelFor(const SimpleForLoopRanges<RankValue>& loop_ranges,
                   const ForLoopRunInfo& run_info,
                   const LambdaType& lambda_function,
                   const ReducerArgs&... reducer_args)
{
  ComplexForLoopRanges<RankValue> complex_loop_ranges{ loop_ranges };
  arccoreParallelFor(complex_loop_ranges, run_info, lambda_function, reducer_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the lambda function \a lambda_function concurrently
 * over the iteration interval given by \a loop_ranges.
 */
template <int RankValue, typename LambdaType, typename... ReducerArgs> inline void
arccoreParallelFor(const SimpleForLoopRanges<RankValue>& loop_ranges,
                   const ParallelLoopOptions& options,
                   const LambdaType& lambda_function,
                   const ReducerArgs&... reducer_args)
{
  ComplexForLoopRanges<RankValue> complex_loop_ranges{ loop_ranges };
  arccoreParallelFor(complex_loop_ranges, ForLoopRunInfo(options), lambda_function, reducer_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the lambda function \a lambda_function concurrently
 * over the iteration interval given by \a loop_ranges.
 */
template <int RankValue, typename LambdaType> inline void
arccoreParallelFor(const ComplexForLoopRanges<RankValue>& loop_ranges,
                   const LambdaType& lambda_function)
{
  ParallelLoopOptions options;
  arccoreParallelFor(loop_ranges, options, lambda_function);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the lambda function \a lambda_function concurrently
 * over the iteration interval given by \a loop_ranges.
 */
template <int RankValue, typename LambdaType> inline void
arccoreParallelFor(const SimpleForLoopRanges<RankValue>& loop_ranges,
                   const LambdaType& lambda_function)
{
  ParallelLoopOptions options;
  ComplexForLoopRanges<RankValue> complex_loop_ranges{ loop_ranges };
  arccoreParallelFor(complex_loop_ranges, options, lambda_function);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the lambda function \a lambda_function concurrently
 * over the iteration interval [i0,i0+size] with the options \a options.
 */
template <typename LambdaType> inline void
arccoreParallelFor(Integer i0, Integer size, const ForLoopRunInfo& options,
                   const LambdaType& lambda_function)
{
  LambdaRangeFunctorT<LambdaType> ipf(lambda_function);
  ParallelFor1DLoopInfo loop_info(i0, size, &ipf, options);
  TaskFactory::executeParallelFor(loop_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
