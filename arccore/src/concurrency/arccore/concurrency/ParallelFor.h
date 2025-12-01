// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelFor.h                                               (C) 2000-2025 */
/*                                                                           */
/* Gestion des boucles parallèles.                                           */
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
 * \brief Caractéristiques d'un boucle 1D multi-thread.
 *
 * Cette classe permet de spécifier les options d'une boucle à paralléliser
 * en mode multi-thread.
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
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération donné par \a loop_ranges.
 */
template <int RankValue, typename LambdaType, typename... ReducerArgs> inline void
arccoreParallelFor(const ComplexForLoopRanges<RankValue>& loop_ranges,
                   const ForLoopRunInfo& run_info,
                   const LambdaType& lambda_function,
                   const ReducerArgs&... reducer_args)
{
  // Modif Arcane 3.7.9 (septembre 2022)
  // Effectue une copie pour privatiser au thread courant les valeurs de la lambda.
  // Cela est nécessaire pour que objets comme les reducers soient bien pris
  // en compte.
  // TODO: regarder si on pourrait faire la copie uniquement une fois par thread
  // si cette copie devient couteuse.
  // NOTE: A partir de la version 3.12.15 (avril 2024), avec la nouvelle version
  // des réducteurs (Reduce2), cette privatisation n'est plus utile. Une fois
  // qu'on aura supprimer les anciennes classes gérant les réductions (Reduce),
  // on pourra supprimer cette privatisation
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
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération donné par \a loop_ranges.
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
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération donné par \a loop_ranges.
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
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération donné par \a loop_ranges.
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
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération donné par \a loop_ranges.
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
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération donné par \a loop_ranges.
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
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération [i0,i0+size] avec les options \a options.
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
