// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshInternal.h                                             (C) 2000-2024 */
/*                                                                           */
/* Partie interne à Arcane de IMesh.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IMESHINTERNAL_H
#define ARCANE_CORE_INTERNAL_IMESHINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class IItemConnectivityMng;
class IPolyhedralMeshModifier;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Partie interne de IMesh.
 */
class ARCANE_CORE_EXPORT IMeshInternal
{
 public:

  virtual ~IMeshInternal() = default;

 public:

  /*!
   * \brief Positionne le type de maillage.
   *
   * Pour l'instant il ne faut utiliser cette méthode que pour spécifier
   * la structure du maillage (eMeshStructure).
   */
  virtual void setMeshKind(const MeshKind& v) = 0;

  /*!
   * \brief Renvoie le gestionnaire de connectivités des dofs.
   *
   * Cette méthode est temporaire car ce gestionnaire de connectivités des dofs
   * à vocation à disparaître, l'évolution des connectivités des dofs étant maintenant gérée
   * automatiquement. A usage interne uniquement en attendant la suppression.
   */
  virtual IItemConnectivityMng* dofConnectivityMng() const noexcept = 0;

  /*!
   * \bief Renvoie l'interface de modification du maillage polyédrique
   *
   * Cette méthode retourne nullptr si l'implémentation du maillage n'est pas PolyhedralMesh
   */
  virtual IPolyhedralMeshModifier* polyhedralMeshModifier() const noexcept = 0;
};



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
