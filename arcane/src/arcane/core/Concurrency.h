// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Concurrency.h                                               (C) 2000-2025 */
/*                                                                           */
/* Classes managing concurrency (tasks, parallel loops, ...)                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CONCURRENCY_H
#define ARCANE_CORE_CONCURRENCY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ConcurrencyUtils.h"

#include "arcane/core/Item.h"
#include "arcane/core/ItemFunctor.h"
#include "arcane/core/ItemGroup.h"

#include "arcane/core/materials/MatItem.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{
inline Int32
adaptGrainSize(const ForLoopRunInfo& run_info)
{
  const std::optional<ParallelLoopOptions>& options = run_info.options();
  Int32 grain_size = AbstractItemRangeFunctor::DEFAULT_GRAIN_SIZE;
  if (options.has_value())
    if (options.value().hasGrainSize())
      grain_size = options.value().grainSize();
  return grain_size;
}
} // namespace Arcane::impl

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Concurrency
 *
 * \brief Applies the method \a function of the instance
 * \a instance concurrently on the view \a items_view with the options \a options.
 */
template <typename InstanceType, typename ItemType> inline void
arcaneParallelForeach(const ItemVectorView& items_view, const ForLoopRunInfo& run_info,
                      InstanceType* instance, void (InstanceType::*function)(ItemVectorViewT<ItemType> items))
{
  Int32 grain_size = impl::adaptGrainSize(run_info);
  ItemRangeFunctorT<InstanceType, ItemType> ipf(items_view, instance, function, grain_size);

  ForLoopRunInfo adapted_run_info(run_info);
  ParallelLoopOptions loop_opt(run_info.options().value_or(TaskFactory::defaultParallelLoopOptions()));
  loop_opt.setGrainSize(ipf.blockGrainSize());
  adapted_run_info.addOptions(loop_opt);

  ParallelFor1DLoopInfo loop_info(0, ipf.nbBlock(), &ipf, adapted_run_info);
  TaskFactory::executeParallelFor(loop_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the lambda function \a lambda_function
 * \a instance concurrently on the view \a items_view with the options \a options
 * \ingroup Concurrency
 */
template <typename LambdaType> inline void
arcaneParallelForeach(const ItemVectorView& items_view, const ForLoopRunInfo& run_info,
                      const LambdaType& lambda_function)
{
  Int32 grain_size = impl::adaptGrainSize(run_info);
  LambdaItemRangeFunctorT<LambdaType> ipf(items_view, lambda_function, grain_size);

  ForLoopRunInfo adapted_run_info(run_info);
  ParallelLoopOptions loop_opt(run_info.options().value_or(TaskFactory::defaultParallelLoopOptions()));
  loop_opt.setGrainSize(ipf.blockGrainSize());
  adapted_run_info.addOptions(loop_opt);

  ParallelFor1DLoopInfo loop_info(0, ipf.nbBlock(), &ipf, adapted_run_info);
  TaskFactory::executeParallelFor(loop_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Concurrency
 *
 * \brief Applies the method \a function of the instance
 * \a instance concurrently on the view \a items_view with the options \a options.
 */
template <typename InstanceType, typename ItemType> inline void
arcaneParallelForeach(const ItemVectorView& items_view, const ParallelLoopOptions& options,
                      InstanceType* instance, void (InstanceType::*function)(ItemVectorViewT<ItemType> items))
{
  arcaneParallelForeach(items_view, ForLoopRunInfo(options), instance, function);
}

/*!
 * \ingroup Concurrency
 *
 * \brief Applies the method \a function of the instance
 * \a instance concurrently on the group \a items with the options \a options.
 */
template <typename InstanceType, typename ItemType> inline void
arcaneParallelForeach(const ItemGroup& items, const ForLoopRunInfo& run_info,
                      InstanceType* instance, void (InstanceType::*function)(ItemVectorViewT<ItemType> items))
{
  arcaneParallelForeach(items._paddedView(), run_info, instance, function);
}

/*!
 * \ingroup Concurrency
 * \brief Applies the method \a function of the instance
 * \a instance concurrently on the view \a items_view.
 */
template <typename InstanceType, typename ItemType> inline void
arcaneParallelForeach(const ItemVectorView& items_view,
                      InstanceType* instance, void (InstanceType::*function)(ItemVectorViewT<ItemType> items))
{
  arcaneParallelForeach(items_view, ForLoopRunInfo(), instance, function);
}

/*!
 * \ingroup Concurrency
 * \brief Applies the method \a function of the instance
 * \a instance concurrently on the group \a items.
 */
template <typename InstanceType, typename ItemType> inline void
arcaneParallelForeach(const ItemGroup& items,
                      InstanceType* instance, void (InstanceType::*function)(ItemVectorViewT<ItemType> items))
{
  arcaneParallelForeach(items._paddedView(), ForLoopRunInfo(), instance, function);
}

/*!
 * \brief Applies the lambda function \a lambda_function
 * \a instance concurrently on the view \a items_view with the options \a options
 * \ingroup Concurrency
 */
template <typename LambdaType> inline void
arcaneParallelForeach(const ItemVectorView& items_view, const ParallelLoopOptions& options,
                      const LambdaType& lambda_function)
{
  arcaneParallelForeach(items_view, ForLoopRunInfo(options), lambda_function);
}

/*!
 * \brief Applies the lambda function \a lambda_function
 * concurrently on the group \a items with the options \a options
 * \ingroup Concurrency
 */
template <typename LambdaType> inline void
arcaneParallelForeach(const ItemGroup& items, const ParallelLoopOptions& options,
                      const LambdaType& lambda_function)
{
  arcaneParallelForeach(items._paddedView(), ForLoopRunInfo(options), lambda_function);
}

/*!
 * \ingroup Concurrency
 * \brief Applies the lambda function \a lambda_function
 * \a instance concurrently on the view \a items_view.
 */
template <typename LambdaType> inline void
arcaneParallelForeach(const ItemVectorView& items_view, const LambdaType& lambda_function)
{
  arcaneParallelForeach(items_view, ForLoopRunInfo(), lambda_function);
}

/*!
 * \ingroup Concurrency
 * \brief Applies the lambda function \a lambda_function
 * concurrently on the group \a items.
 */
template <typename LambdaType> inline void
arcaneParallelForeach(const ItemGroup& items, const LambdaType& lambda_function)
{
  arcaneParallelForeach(items._paddedView(), lambda_function);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Concurrency
 * \brief Applies the lambda function \a lambda_function
 * concurrently over the iteration range [i0,i0+size].
 */
template <typename InstanceType> inline void
arcaneParallelFor(Integer i0, Integer size, InstanceType* itype,
                  void (InstanceType::*lambda_function)(Integer i0, Integer size))
{
  RangeFunctorT<InstanceType> ipf(itype, lambda_function);
  ParallelFor1DLoopInfo loop_info(i0, size, &ipf);
  TaskFactory::executeParallelFor(loop_info);
}

/*!
 * \ingroup Concurrency
 * \brief Applies the lambda function \a lambda_function
 * concurrently over the iteration range [i0,i0+size] with the options \a options.
 */
template <typename LambdaType> inline void
arcaneParallelFor(Integer i0, Integer size, const ForLoopRunInfo& options,
                  const LambdaType& lambda_function)
{
  LambdaRangeFunctorT<LambdaType> ipf(lambda_function);
  ParallelFor1DLoopInfo loop_info(i0, size, &ipf, options);
  TaskFactory::executeParallelFor(loop_info);
}

/*!
 * \ingroup Concurrency
 * \brief Applies the lambda function \a lambda_function
 * concurrently over the iteration range [i0,i0+size] with the options \a options.
 */
template <typename LambdaType> inline void
arcaneParallelFor(Integer i0, Integer size, const ParallelLoopOptions& options,
                  const LambdaType& lambda_function)
{
  arcaneParallelFor(i0, size, ForLoopRunInfo(options), lambda_function);
}

/*!
 * \brief Applies the lambda function \a lambda_function
 * concurrently over the iteration range [i0,i0+size]
 */
template <typename LambdaType> inline void
arcaneParallelFor(Integer i0, Integer size, const LambdaType& lambda_function)
{
  LambdaRangeFunctorT<LambdaType> ipf(lambda_function);
  ParallelFor1DLoopInfo loop_info(i0, size, &ipf);
  TaskFactory::executeParallelFor(loop_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the lambda function \a lambda_function
 * \a instance concurrently on the container views \a views with the options \a options
 * \ingroup Concurrency
 */
template <typename LambdaType, typename... Views> inline void
arcaneParallelForVa(const ForLoopRunInfo& run_info, const LambdaType& lambda_function, Views... views)
{
  // Asserting every views have the size
  typename std::tuple_element_t<0, std::tuple<Views...>>::size_type sizes[] = { views.size()... };
  if (!std::all_of(std::begin(sizes), std::end(sizes), [&sizes](auto cur) { return cur == sizes[0]; }))
    ARCANE_FATAL("Every views must have the same size");

  LambdaRangeFunctorTVa<LambdaType, Views...> ipf(views..., lambda_function);

  ParallelFor1DLoopInfo loop_info(0, sizes[0], &ipf, run_info);
  TaskFactory::executeParallelFor(loop_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Concurrency implementation.
 * * The methods in this namespace are obsolete and must be replaced
 * by equivalent methods in the Arcane namespace.
 * For example, Arcane::Parallel::For() must be replaced by Arcane::arcaneParallelFor()
 * and Arcane::Parallel::Foreach() by Arcane::arcaneParallelForeach().
 */
namespace Parallel
{
  /*!
   * \deprecated Use Arcane::arcaneParallelForeach() instead.
   */
  template <typename InstanceType, typename ItemType>
  [[deprecated("Year2021: Use Arcane::arcaneParallelForeach() instead")]] inline void
  Foreach(const ItemVectorView& items_view, const ParallelLoopOptions& options,
          InstanceType* instance, void (InstanceType::*function)(ItemVectorViewT<ItemType> items))
  {
    ItemRangeFunctorT<InstanceType, ItemType> ipf(items_view, instance, function, options.grainSize());
    // Recopie \a options et utilise la valeur de 'grain_size' retournée par \a ifp
    ParallelLoopOptions loop_opt(options);
    loop_opt.setGrainSize(ipf.blockGrainSize());
    TaskFactory::executeParallelFor(0, ipf.nbBlock(), loop_opt, &ipf);
  }

  /*!
   * \deprecated Use Arcane::arcaneParallelForeach() instead.
   */
  template <typename InstanceType, typename ItemType>
  [[deprecated("Year2021: Use Arcane::arcaneParallelForeach() instead")]] inline void
  Foreach(const ItemGroup& items, const ParallelLoopOptions& options, InstanceType* instance,
          void (InstanceType::*function)(ItemVectorViewT<ItemType> items))
  {
    Foreach(items._paddedView(), options, instance, function);
  }

  /*!
   * \deprecated Use Arcane::arcaneParallelForeach() instead.
   */
  template <typename InstanceType, typename ItemType>
  [[deprecated("Year2021: Use Arcane::arcaneParallelForeach() instead")]] inline void
  Foreach(const ItemVectorView& items_view, InstanceType* instance, void (InstanceType::*function)(ItemVectorViewT<ItemType> items))
  {
    ItemRangeFunctorT<InstanceType, ItemType> ipf(items_view, instance, function);
    TaskFactory::executeParallelFor(0, ipf.nbBlock(), ipf.blockGrainSize(), &ipf);
  }

  /*!
   * \deprecated Use Arcane::arcaneParallelForeach() instead.
   */
  template <typename InstanceType, typename ItemType>
  [[deprecated("Year2021: Use Arcane::arcaneParallelForeach() instead")]] inline void
  Foreach(const ItemGroup& items, InstanceType* instance, void (InstanceType::*function)(ItemVectorViewT<ItemType> items))
  {
    Foreach(items._paddedView(), instance, function);
  }

  /*!
   * \deprecated Use Arcane::arcaneParallelForeach() instead.
   */
  template <typename LambdaType>
  [[deprecated("Year2021: Use Arcane::arcaneParallelForeach() instead")]] inline void
  Foreach(const ItemVectorView& items_view, const ParallelLoopOptions& options, const LambdaType& lambda_function)
  {
    LambdaItemRangeFunctorT<LambdaType> ipf(items_view, lambda_function, options.grainSize());
    // Recopie \a options et utilise la valeur de 'grain_size' retournée par \a ifp
    ParallelLoopOptions loop_opt(options);
    loop_opt.setGrainSize(ipf.blockGrainSize());
    TaskFactory::executeParallelFor(0, ipf.nbBlock(), loop_opt, &ipf);
  }

  /*!
   * \deprecated Use Arcane::arcaneParallelForeach() instead.
   */
  template <typename LambdaType>
  [[deprecated("Year2021: Use Arcane::arcaneParallelForeach() instead")]] inline void
  Foreach(const ItemGroup& items, const ParallelLoopOptions& options, const LambdaType& lambda_function)
  {
    Foreach(items._paddedView(), options, lambda_function);
  }

  /*!
   * \deprecated Use Arcane::arcaneParallelForeach() instead.
   */
  template <typename LambdaType>
  [[deprecated("Year2021: Use Arcane::arcaneParallelForeach() instead")]] inline void
  Foreach(const ItemVectorView& items_view, const LambdaType& lambda_function)
  {
    LambdaItemRangeFunctorT<LambdaType> ipf(items_view, lambda_function);
    TaskFactory::executeParallelFor(0, ipf.nbBlock(), ipf.blockGrainSize(), &ipf);
  }

  /*!
   * \deprecated Use Arcane::arcaneParallelForeach() instead.
   */
  template <typename LambdaType>
  [[deprecated("Year2021: Use Arcane::arcaneParallelForeach() instead")]] inline void
  Foreach(const ItemGroup& items, const LambdaType& lambda_function)
  {
    Foreach(items._paddedView(), lambda_function);
  }

  /*!
   * \deprecated Use the For overload with ParallelLoopOptions as an argument.
   */
  template <typename InstanceType> ARCANE_DEPRECATED_122 inline void
  For(Integer i0, Integer size, Integer grain_size, InstanceType* itype,
      void (InstanceType::*lambda_function)(Integer i0, Integer size))
  {
    RangeFunctorT<InstanceType> ipf(itype, lambda_function);
    TaskFactory::executeParallelFor(i0, size, grain_size, &ipf);
  }

  /*!
   * \deprecated Use Arcane::arcaneParallelFor() instead.
   */
  template <typename InstanceType>
  [[deprecated("Year2021: Use Arcane::arcaneParallelFor() instead")]] inline void
  For(Integer i0, Integer size, const ParallelLoopOptions& options, InstanceType* itype,
      void (InstanceType::*lambda_function)(Integer i0, Integer size))
  {
    RangeFunctorT<InstanceType> ipf(itype, lambda_function);
    TaskFactory::executeParallelFor(i0, size, options, &ipf);
  }

  /*!
   * \deprecated Use the For overload with ParallelLoopOptions as an argument.
   */
  template <typename LambdaType> ARCANE_DEPRECATED_122 inline void
  For(Integer i0, Integer size, Integer grain_size, const LambdaType& lambda_function)
  {
    LambdaRangeFunctorT<LambdaType> ipf(lambda_function);
    TaskFactory::executeParallelFor(i0, size, grain_size, &ipf);
  }

  /*!
   * \deprecated Use Arcane::arcaneParallelFor() instead.
   */
  template <typename InstanceType>
  [[deprecated("Year2021: Use Arcane::arcaneParallelFor() instead")]] inline void
  For(Integer i0, Integer size, InstanceType* itype,
      void (InstanceType::*lambda_function)(Integer i0, Integer size))
  {
    RangeFunctorT<InstanceType> ipf(itype, lambda_function);
    TaskFactory::executeParallelFor(i0, size, &ipf);
  }

  /*!
   * \deprecated Use Arcane::arcaneParallelFor() instead.
   */
  template <typename LambdaType>
  [[deprecated("Year2021: Use Arcane::arcaneParallelFor() instead")]] inline void
  For(Integer i0, Integer size, const ParallelLoopOptions& options, const LambdaType& lambda_function)
  {
    LambdaRangeFunctorT<LambdaType> ipf(lambda_function);
    TaskFactory::executeParallelFor(i0, size, options, &ipf);
  }

  /*!
   * \deprecated Use Arcane::arcaneParallelFor() instead.
   */
  template <typename LambdaType>
  [[deprecated("Year2021: Use Arcane::arcaneParallelFor() instead")]] inline void
  For(Integer i0, Integer size, const LambdaType& lambda_function)
  {
    LambdaRangeFunctorT<LambdaType> ipf(lambda_function);
    TaskFactory::executeParallelFor(i0, size, &ipf);
  }

} // End namespace Parallel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
