// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilyInternal.h                                       (C) 2000-2025 */
/*                                                                           */
/* Partie interne à Arcane de IItemFamily.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IITEMFAMILYINTERNAL_H
#define ARCANE_CORE_INTERNAL_IITEMFAMILYINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IItemFamilyTopologyModifier;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Partie interne de IItemFamily.
 */
class ARCANE_CORE_EXPORT IItemFamilyInternal
{
 public:

  virtual ~IItemFamilyInternal() = default;

 public:

  //! Informations sur les connectivités non structurés
  virtual ItemInternalConnectivityList* unstructuredItemInternalConnectivityList() = 0;

  //! Interface du modificateur de topologie.
  virtual IItemFamilyTopologyModifier* topologyModifier() = 0;

  //! Instance de ItemSharedInfo pour les entités de la famille
  virtual ItemSharedInfo* commonItemSharedInfo() = 0;

  /*!
   * \brief Indique une fin d'allocation.
   *
   * Cette méthode ne doit normalement être appelée que par le maillage (IMesh)
   * au moment de l'allocate.
   *
   * Cette méthode est collective.
   */
  virtual void endAllocate() = 0;

  /*!
   * \brief Indique une fin de modification par le maillage.
   *
   * Cette méthode ne doit normalement être appelée que par le maillage (IMesh)
   * à la fin d'un endUpdate().
   *
   * Cette méthode est collective.
   */
  virtual void notifyEndUpdateFromMesh() = 0;

  /*!
   * \brief Ajoute une variable à cette famille.
   *
   * Cette méthode est appelée par la variable elle même et ne doit pas
   * être apelée dans d'autres conditions.
   */
  virtual void addVariable(IVariable* var) = 0;

  /*!
   * \brief Supprime une variable à cette famille.
   *
   * Cette méthode est appelée par la variable elle même et ne doit pas
   * être apelée dans d'autres conditions.
   */
  virtual void removeVariable(IVariable* var) = 0;

  /*!
   * \brief Redimensionne les variables de cette famille.
   */
  virtual void resizeVariables(bool force_resize) = 0;

  virtual void addSourceConnectivity(IIncrementalItemSourceConnectivity* connectivity) = 0;
  virtual void addTargetConnectivity(IIncrementalItemTargetConnectivity* connectivity) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
