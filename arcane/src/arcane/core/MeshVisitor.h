// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVisitor.h                                               (C) 2000-2016 */
/*                                                                           */
/* Visiteurs divers sur les entités du maillage.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHVISITOR_H
#define ARCANE_MESHVISITOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FunctorUtils.h"

#include "arcane/ArcaneTypes.h"
#include "arcane/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Namespace contenant un ensemble de visiteurs liés au maillage.
 *
 * Par exemple pour visiter l'ensemble des groupes d'une famille:
 \code
 * IItemFamily* f = ...;
 * auto xx = [](const ItemGroup& x) { std::cout << "name=" << x.name(); };
 * MeshVisitor::visitGroups(f,xx);
 \endcode


 * ou pour l'ensemble des groupes de l'ensemble des familles:

 \code
 * IMesh* mesh = ...;
 * auto xx = [](const ItemGroup& x) { std::cout << "name=" << x.name(); };
 * MeshVisitor::visitGroups(mesh,xx);
 \endcode

 */
namespace meshvisitor
{
/*!
 * \brief Visite l'ensemble des groupes de \a family avec le functor \a functor.
 */
ARCANE_CORE_EXPORT void 
visitGroups(IItemFamily* family,IFunctorWithArgumentT<ItemGroup&>* functor);

/*!
 * \brief Visite l'ensemble des groupes de \a mesh avec le functor \a functor.
 */
ARCANE_CORE_EXPORT void 
visitGroups(IMesh* mesh,IFunctorWithArgumentT<ItemGroup&>* functor);

/*!
 * \brief Visite l'ensemble des groupes de \a family avec la lambda \a f.
 */
template<typename LambdaType> inline void
visitGroups(IItemFamily* family,const LambdaType& f)
{
  StdFunctorWithArgumentT<ItemGroup&> sf(f);
  // Il faut caster en le bon type que le compilateur utilise la bonne surcharge.
  IFunctorWithArgumentT<ItemGroup&>* sf_addr = &sf;
  visitGroups(family,sf_addr);
}
/*!
 * \brief Visite l'ensemble des groupes de \a mesh avec la lambda \a f.
 */
template<typename LambdaType> inline void
visitGroups(IMesh* mesh,const LambdaType& f)
{
  StdFunctorWithArgumentT<ItemGroup&> sf(f);
  // Il faut caster en le bon type que le compilateur utilise la bonne surcharge.
  IFunctorWithArgumentT<ItemGroup&>* sf_addr = &sf;
  visitGroups(mesh,sf_addr);
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

