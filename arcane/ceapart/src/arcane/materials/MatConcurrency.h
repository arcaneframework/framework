// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatConcurrency.h                                            (C) 2000-2021 */
/*                                                                           */
/* Classes gérant la concurrence pour les matériaux.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MATCONCURRENCY_H
#define ARCANE_MATERIALS_MATCONCURRENCY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/RangeFunctor.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/Concurrency.h"

#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/ComponentItemVectorView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctor sur un interval d'itération instancié via une lambda fonction.
 *
 * Le type \a ViewType doit être choisi parmi ComponentItemVectorView,
 * MatItemVectorView ou EnvItemVectorView.
 */
template<typename ViewType,typename LambdaType>
class LambdaMatItemRangeFunctorT
: public IRangeFunctor
{
 public:
  LambdaMatItemRangeFunctorT(ViewType items_view,const LambdaType& lambda_function)
  : m_items(items_view), m_lambda_function(lambda_function)
  {
  }
 
 public:
  
  virtual void executeFunctor(Integer begin,Integer size)
  {
    ViewType sub_view(m_items._subView(begin,size));
    m_lambda_function(sub_view);
  }
 
 private:
  ViewType m_items;
  const LambdaType& m_lambda_function;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
  using namespace Materials;
  /*! 
   * \brief Applique en concurrence la fonction lambda \a lambda_function
   * \a instance sur la vue du composant \a items_view.
   * \ingroup Concurrency
   */
  template<typename LambdaType> inline void
  arcaneParallelForeach(const ComponentItemVectorView& items_view,const LambdaType& lambda_function)
  {
    LambdaMatItemRangeFunctorT<ComponentItemVectorView,LambdaType> ipf(items_view,lambda_function);
    TaskFactory::executeParallelFor(0,items_view.nbItem(),&ipf);
  }

  /*! 
   * \brief Applique en concurrence la fonction lambda \a lambda_function
   * \a instance sur la vue du composant \a items_view avec les options \a options.
   * \ingroup Concurrency
   */
  template<typename LambdaType> inline void
  arcaneParallelForeach(const ComponentItemVectorView& items_view,const ParallelLoopOptions& options,
                       const LambdaType& lambda_function)
  {
    LambdaMatItemRangeFunctorT<ComponentItemVectorView,LambdaType> ipf(items_view,lambda_function);
    TaskFactory::executeParallelFor(0,items_view.nbItem(),options,&ipf);
  }

  /*!
   * \brief Applique en concurrence la fonction lambda \a lambda_function
   * \a instance sur la vue milieux \a items_view.
   * \ingroup Concurrency
   */
  template<typename LambdaType> inline void
  arcaneParallelForeach(const EnvItemVectorView& items_view,const LambdaType& lambda_function)
  {
    LambdaMatItemRangeFunctorT<EnvItemVectorView,LambdaType> ipf(items_view,lambda_function);
    TaskFactory::executeParallelFor(0,items_view.nbItem(),&ipf);
  }

  /*! 
   * \brief Applique en concurrence la fonction lambda \a lambda_function
   * \a instance sur la vue milieux \a items_view avec les options \a options.
   * \ingroup Concurrency
   */
  template<typename LambdaType> inline void
  arcaneParallelForeach(const EnvItemVectorView& items_view,const ParallelLoopOptions& options,
                       const LambdaType& lambda_function)
  {
    LambdaMatItemRangeFunctorT<EnvItemVectorView,LambdaType> ipf(items_view,lambda_function);
    TaskFactory::executeParallelFor(0,items_view.nbItem(),options,&ipf);
  }

  /*!
   * \brief Applique en concurrence la fonction lambda \a lambda_function
   * \a instance sur la vue matériaux \a items_view.
   * \ingroup Concurrency
   */
  template<typename LambdaType> inline void
  arcaneParallelForeach(const MatItemVectorView& items_view,const LambdaType& lambda_function)
  {
    LambdaMatItemRangeFunctorT<MatItemVectorView,LambdaType> ipf(items_view,lambda_function);
    TaskFactory::executeParallelFor(0,items_view.nbItem(),&ipf);
  }

  /*! 
   * \brief Applique en concurrence la fonction lambda \a lambda_function
   * \a instance sur la vue matériaux \a items_view avec les options \a options.
   * \ingroup Concurrency
   */
  template<typename LambdaType> inline void
  arcaneParallelForeach(const MatItemVectorView& items_view,const ParallelLoopOptions& options,
          const LambdaType& lambda_function)
  {
    LambdaMatItemRangeFunctorT<MatItemVectorView,LambdaType> ipf(items_view,lambda_function);
    TaskFactory::executeParallelFor(0,items_view.nbItem(),options,&ipf);
  }
} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Parallel
{
  using namespace Materials;
  using namespace Arcane;

  /*! 
   * \brief Applique en concurrence la fonction lambda \a lambda_function
   * \a instance sur la vue du composant \a items_view.
   * \ingroup Concurrency
   */
  template<typename LambdaType> inline void
  Foreach(const ComponentItemVectorView& items_view,const LambdaType& lambda_function)
  {
    arcaneParallelForeach(items_view,lambda_function);
  }

  /*! 
   * \brief Applique en concurrence la fonction lambda \a lambda_function
   * \a instance sur la vue du composant \a items_view avec les options \a options.
   * \ingroup Concurrency
   */
  template<typename LambdaType> inline void
  Foreach(const ComponentItemVectorView& items_view,const ParallelLoopOptions& options,
          const LambdaType& lambda_function)
  {
    arcaneParallelForeach(items_view,options,lambda_function);
  }

  /*!
   * \brief Applique en concurrence la fonction lambda \a lambda_function
   * \a instance sur la vue milieux \a items_view.
   * \ingroup Concurrency
   */
  template<typename LambdaType> inline void
  Foreach(const EnvItemVectorView& items_view,const LambdaType& lambda_function)
  {
    arcaneParallelForeach(items_view,lambda_function);
  }

  /*! 
   * \brief Applique en concurrence la fonction lambda \a lambda_function
   * \a instance sur la vue milieux \a items_view avec les options \a options.
   * \ingroup Concurrency
   */
  template<typename LambdaType> inline void
  Foreach(const EnvItemVectorView& items_view,const ParallelLoopOptions& options,
          const LambdaType& lambda_function)
  {
    arcaneParallelForeach(items_view,options,lambda_function);
  }

  /*!
   * \brief Applique en concurrence la fonction lambda \a lambda_function
   * \a instance sur la vue matériaux \a items_view.
   * \ingroup Concurrency
   */
  template<typename LambdaType> inline void
  Foreach(const MatItemVectorView& items_view,const LambdaType& lambda_function)
  {
    arcaneParallelForeach(items_view,lambda_function);
  }

  /*! 
   * \brief Applique en concurrence la fonction lambda \a lambda_function
   * \a instance sur la vue matériaux \a items_view avec les options \a options.
   * \ingroup Concurrency
   */
  template<typename LambdaType> inline void
  Foreach(const MatItemVectorView& items_view,const ParallelLoopOptions& options,
          const LambdaType& lambda_function)
  {
    arcaneParallelForeach(items_view,options,lambda_function);
  }

} // End namespace Arcane::Parallel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

