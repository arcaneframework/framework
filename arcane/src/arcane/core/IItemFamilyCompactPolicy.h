// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilyCompactPolicy.h                                  (C) 2000-2025 */
/*                                                                           */
/* Interface de la politique de compactage des entités d'une famille.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMFAMILYCOMPACTPOLICY_H
#define ARCANE_CORE_IITEMFAMILYCOMPACTPOLICY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Politique de compactage des entités.
 *
 * Une instance de cette classe est associée à chaque famille.
 *
 * Le pseudo-code d'appel pour un compactage est le suivant:
 *
 \code
 * IMesh* mesh = ...;
 * IMeshCompacter* compacter = ...;
 * ItemFamilyCollection families = mesh->itemFamilies();
 * UniqueArray<ItemFamilyCompactPolicy> policies;
 * for( IItemFamily* family : mesh->itemFamilies() )
 *   policies.add( createCompactPolicity(family) );
 * for( ItemFamilyCompactPolicity* policy : policies)
 *   policy->beginCompact(...);
 * for( ItemFamilyCompactPolicity* policy : policies)
 *   policy->compactVariablesAndGroups(...);
 * for( ItemFamilyCompactPolicity* policy : policies)
 *   policy->updateInternalReferences(compacter);
 * for( ItemFamilyCompactPolicity* policy : policies)
 *   policy->endCompact(...);
 \endcode
 *
 * En dehors d'un compactage, il est possible d'appeler
 * compactReferenceData() qui permet de compacter les données servant
 * à contenir les infos de connectivité.
 * données.
 */
class ARCANE_CORE_EXPORT IItemFamilyCompactPolicy
{
 public:

  virtual ~IItemFamilyCompactPolicy() = default;

 public:

  // TODO: faire une méthode computeCompact après beginCompact().
  virtual void beginCompact(ItemFamilyCompactInfos& compact_infos) = 0;
  virtual void compactVariablesAndGroups(const ItemFamilyCompactInfos& compact_infos) = 0;
  virtual void updateInternalReferences(IMeshCompacter* compacter) = 0;
  virtual void endCompact(ItemFamilyCompactInfos& compact_infos) = 0;
  virtual void finalizeCompact(IMeshCompacter* compacter) = 0;
  //! Famille associée
  virtual IItemFamily* family() const = 0;
  //! Compacte les données sur les connectivités.
  virtual void compactConnectivityData() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
