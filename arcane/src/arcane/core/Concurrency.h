// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Concurrency.h                                               (C) 2000-2022 */
/*                                                                           */
/* Classes gérant la concurrence (tâches, boucles parallèles, ...)           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CONCURRENCY_H
#define ARCANE_CONCURRENCY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ConcurrencyUtils.h"

#include "arcane/Item.h"
#include "arcane/ItemFunctor.h"
#include "arcane/ItemGroup.h"

#include <arcane/core/materials/MatItem.h>

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
 * \brief Applique en concurrence la méthode \a function de l'instance
 * \a instance sur la vue \a items_view avec les options \a options.
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
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * \a instance sur la vue \a items_view avec les options \a options
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
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * \a instance sur les vues des containers \a items_view avec les options \a options
 * \ingroup Concurrency
 */
template <typename LambdaType, typename... Views> inline void
arcaneParallelForeach_FL(const ForLoopRunInfo& run_info, const LambdaType& lambda_function, Views... views)
{
  Int32 grain_size = impl::adaptGrainSize(run_info);

  // Asserting every views have the size
  // TODO: must be encapsulated in some king of check or debug mode
  typename FirstVariadicType<Views...>::size_type sizes[] = {views.size()...};
  if (!std::all_of(std::begin(sizes), std::end(sizes),[&sizes](auto cur){return cur == sizes[0];}))
    ARCANE_FATAL("Every views must have the same size");

  LambdaItemRangeFunctorT_FL<LambdaType, Views...> ipf(views..., lambda_function, grain_size);

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
 * \brief Applique en concurrence la méthode \a function de l'instance
 * \a instance sur la vue \a items_view avec les options \a options.
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
 * \brief Applique en concurrence la méthode \a function de l'instance
 * \a instance sur le groupe \a items avec les options \a options.
 */
template <typename InstanceType, typename ItemType> inline void
arcaneParallelForeach(const ItemGroup& items, const ForLoopRunInfo& run_info,
                      InstanceType* instance, void (InstanceType::*function)(ItemVectorViewT<ItemType> items))
{
  arcaneParallelForeach(items.view(), run_info, instance, function);
}

/*!
 * \ingroup Concurrency
 * \brief Applique en concurrence la méthode \a function de l'instance
 * \a instance sur la vue \a items_view.
 */
template <typename InstanceType, typename ItemType> inline void
arcaneParallelForeach(const ItemVectorView& items_view,
                      InstanceType* instance, void (InstanceType::*function)(ItemVectorViewT<ItemType> items))
{
  arcaneParallelForeach(items_view, ForLoopRunInfo(), instance, function);
}

/*!
 * \ingroup Concurrency
 * \brief Applique en concurrence la méthode \a function de l'instance
 * \a instance sur le groupe \a items.
 */
template <typename InstanceType, typename ItemType> inline void
arcaneParallelForeach(const ItemGroup& items,
                      InstanceType* instance, void (InstanceType::*function)(ItemVectorViewT<ItemType> items))
{
  arcaneParallelForeach(items.view(), ForLoopRunInfo(), instance, function);
}

/*!
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * \a instance sur la vue \a items_view avec les options \a options
 * \ingroup Concurrency
 */
template <typename LambdaType> inline void
arcaneParallelForeach(const ItemVectorView& items_view, const ParallelLoopOptions& options,
                      const LambdaType& lambda_function)
{
  arcaneParallelForeach(items_view, ForLoopRunInfo(options), lambda_function);
}

/*!
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur le groupe \a items avec les options \a options
 * \ingroup Concurrency
 */
template <typename LambdaType> inline void
arcaneParallelForeach(const ItemGroup& items, const ParallelLoopOptions& options,
                      const LambdaType& lambda_function)
{
  arcaneParallelForeach(items.view(), ForLoopRunInfo(options), lambda_function);
}

/*!
 * \ingroup Concurrency
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * \a instance sur la vue \a items_view.
 */
template <typename LambdaType> inline void
arcaneParallelForeach(const ItemVectorView& items_view, const LambdaType& lambda_function)
{
  arcaneParallelForeach(items_view, ForLoopRunInfo(), lambda_function);
}

/*!
 * \ingroup Concurrency
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur le groupe \a items.
 */
template <typename LambdaType> inline void
arcaneParallelForeach(const ItemGroup& items, const LambdaType& lambda_function)
{
  arcaneParallelForeach(items.view(), lambda_function);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Concurrency
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération [i0,i0+size].
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
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération [i0,i0+size] avec les options \a options.
 */
template <typename LambdaType> inline void
arcaneParallelFor(Integer i0, Integer size, const ParallelLoopOptions& options,
                  const LambdaType& lambda_function)
{
  LambdaRangeFunctorT<LambdaType> ipf(lambda_function);
  ParallelFor1DLoopInfo loop_info(i0, size, &ipf, ForLoopRunInfo(options));
  TaskFactory::executeParallelFor(loop_info);
}

/*!
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération [i0,i0+size]
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
 * \brief Implémentation de la concurrence.
 *
 * Les méthodes de ce namespace sont obsolètes et doivent être remplacées
 * par les méthodes équivalentes dans le namespace Arcane.
 * Par exemple Arcane::Parallel::For() doit être remplacé par Arcane::arcaneParallelFor()
 * et Arcane::Parallel::Foreach() par Arcane::arcaneParallelForeach().
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
    Foreach(items.view(), options, instance, function);
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
    Foreach(items.view(), instance, function);
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
    Foreach(items.view(), options, lambda_function);
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
    Foreach(items.view(), lambda_function);
  }

  /*!
   * \deprecated Utiliser la surcharge For avec ParallelLoopOptions en argument.
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
   * \deprecated Utiliser la surcharge For avec ParallelLoopOptions en argument.
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
