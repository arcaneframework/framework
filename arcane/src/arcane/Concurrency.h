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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
arcaneParallelForeach(const ItemVectorView& items_view, const ParallelLoopOptions& options,
                      InstanceType* instance, void (InstanceType::*function)(ItemVectorViewT<ItemType> items))
{
  ItemRangeFunctorT<InstanceType, ItemType> ipf(items_view, instance, function, options.grainSize());
  // Recopie \a options et utilise la valeur de 'grain_size' retournée par \a ifp
  ParallelLoopOptions loop_opt(options);
  loop_opt.setGrainSize(ipf.blockGrainSize());
  TaskFactory::executeParallelFor(0, ipf.nbBlock(), loop_opt, &ipf);
}

/*!
 * \ingroup Concurrency
 *
 * \brief Applique en concurrence la méthode \a function de l'instance
 * \a instance sur le groupe \a items avec les options \a options.
 */
template <typename InstanceType, typename ItemType> inline void
arcaneParallelForeach(const ItemGroup& items, const ParallelLoopOptions& options, InstanceType* instance,
                      void (InstanceType::*function)(ItemVectorViewT<ItemType> items))
{
  arcaneParallelForeach(items.view(), options, instance, function);
}

/*!
 * \ingroup Concurrency
 * \brief Applique en concurrence la méthode \a function de l'instance
 * \a instance sur la vue \a items_view.
 */
template <typename InstanceType, typename ItemType> inline void
arcaneParallelForeach(const ItemVectorView& items_view, InstanceType* instance,
                      void (InstanceType::*function)(ItemVectorViewT<ItemType> items))
{
  ItemRangeFunctorT<InstanceType, ItemType> ipf(items_view, instance, function);
  TaskFactory::executeParallelFor(0, ipf.nbBlock(), ipf.blockGrainSize(), &ipf);
}

/*!
 * \ingroup Concurrency
 * \brief Applique en concurrence la méthode \a function de l'instance
 * \a instance sur le groupe \a items.
 */
template <typename InstanceType, typename ItemType> inline void
arcaneParallelForeach(const ItemGroup& items, InstanceType* instance,
                      void (InstanceType::*function)(ItemVectorViewT<ItemType> items))
{
  arcaneParallelForeach(items.view(), instance, function);
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
  LambdaItemRangeFunctorT<LambdaType> ipf(items_view, lambda_function, options.grainSize());
  // Recopie \a options et utilise la valeur de 'grain_size' retournée par \a ifp
  ParallelLoopOptions loop_opt(options);
  loop_opt.setGrainSize(ipf.blockGrainSize());
  TaskFactory::executeParallelFor(0, ipf.nbBlock(), loop_opt, &ipf);
}

/*!
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur le groupe \a items avec les options \a options
 * \ingroup Concurrency
 */
template <typename LambdaType> inline void
arcaneParallelForeach(const ItemGroup& items, const ParallelLoopOptions& options, const LambdaType& lambda_function)
{
  arcaneParallelForeach(items.view(), options, lambda_function);
}

/*!
 * \ingroup Concurrency
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * \a instance sur la vue \a items_view.
 */
template <typename LambdaType> inline void
arcaneParallelForeach(const ItemVectorView& items_view, const LambdaType& lambda_function)
{
  LambdaItemRangeFunctorT<LambdaType> ipf(items_view, lambda_function);
  TaskFactory::executeParallelFor(0, ipf.nbBlock(), ipf.blockGrainSize(), &ipf);
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
  TaskFactory::executeParallelFor(i0, size, &ipf);
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
  TaskFactory::executeParallelFor(i0, size, options, &ipf);
}

/*!
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération [i0,i0+size]
 */
template <typename LambdaType> inline void
arcaneParallelFor(Integer i0, Integer size, const LambdaType& lambda_function)
{
  LambdaRangeFunctorT<LambdaType> ipf(lambda_function);
  TaskFactory::executeParallelFor(i0, size, &ipf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
