// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshRenumberingInfo.h                              (C) 2000-2023 */
/*                                                                           */
/* Informations pour la renumérotation.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHRENUMBERINGINFO_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHRENUMBERINGINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/cartesianmesh/CartesianPatch.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*!
 * \brief Informations pour la renumérotation.
 *
 * Si on renumérote à la fois les uniqueId() des faces et des entités des
 * patchs, la renumérotation des faces a lieu en premier.
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianMeshRenumberingInfo
{
 public:

  /*!
   * \brief Méthode pour renuméroter les patchs.
   *
   * Si 0, il n'y a pas de renumérotation. La seule autre valeur valide est 1. Dans ce
   * cas, les uniqueId() des entités (Node,Face,Cell) des patches sont renumérotées
   * pour avoir la même numérotation
   * quel que soit le découpage. La numérotation n'est pas contigue. Seules
   * les entités des patchs sont renumérotées. Les entités issues du maillage initial
   * ne sont pas renumérotées.
   */
  void setRenumberPatchMethod(Int32 v) { m_renumber_patch_method = v; }
  Int32 renumberPatchMethod() const { return m_renumber_patch_method; }

  /*!
   * \brief Méthode pour renuméroter les faces.
   *
   * Si 0, il n'y a pas de renumérotation. La seule autre valeur valide est 1.
   * Dans ce cas la renumérotation se base sur une numérotation cartésienne.
   */
  void setRenumberFaceMethod(Int32 v) { m_renumber_faces_method = v; }
  Int32 renumberFaceMethod() const { return m_renumber_faces_method; }

  /*!
   * \brief Indique si on retrie les entités après renumérotation.
   *
   * Le tri appelle IItemFamily::compactItems(true) pour chaque famille.
   * Cela provoque aussi un appel à ICartesianMesh::computeDirections()
   * pour recalculer les directions qui sont invalidées suite au tri.
   */
  void setSortAfterRenumbering(bool v) { m_is_sort = v; }
  bool isSortAfterRenumbering() const { return m_is_sort; }

  /*!
   * \brief Numéro du patch parent pour la renumérotation.
   *
   * Patch parent pour la renumérotation. Pour la renumérotation, on parcourt
   * récursivement les mailles filles de ce patch et on les renumérote ainsi
   * que les entités liées à ces mailles (noeuds et faces).
   *
   * Les entités de ce patch (mailles, noeuds et faces) ne sont pas renumérotées.
   *
   * Si non spécifié, l'implémentation utilisera le patch 0 comme patch parent.
   *
   * \note Cette propriété n'est prise en compte que si renumberPatchMethod()==1.
   */
  void setParentPatch(CartesianPatch patch) { m_parent_patch = patch; }
  CartesianPatch parentPatch() const { return m_parent_patch; }

 private:

  Int32 m_renumber_patch_method = 0;
  Int32 m_renumber_faces_method = 0;
  CartesianPatch m_parent_patch;
  bool m_is_sort = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

