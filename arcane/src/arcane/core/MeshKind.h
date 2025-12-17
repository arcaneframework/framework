// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshKind.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Caractéristiques d'un maillage.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHKIND_H
#define ARCANE_CORE_MESHKIND_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Structure du maillage
enum class eMeshStructure
{
  //! Structure inconnu ou pas initialisée
  Unknown,
  //! Maillage non structuré
  Unstructured,
  //! Maillage cartésien
  Cartesian,
  //! Maillage polyedrique
  Polyhedral
};

extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& o, eMeshStructure r);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Type de maillage AMR
enum class eMeshAMRKind
{
  //! Le maillage n'est pas AMR
  None,
  //! Le maillage est AMR par maille
  Cell,
  //! Le maillage est AMR par patch
  Patch,
  //! Le maillage est AMR par patch cartésien (rectangulaire)
  PatchCartesianMeshOnly
};

extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& o, eMeshAMRKind r);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Types de gestion de la dimension du maillage.
 *
 * \warning Les modes autres que eMeshCellDimensionKind::MonoDimension sont
 * expérimentaux et ne sont supportés que pour eMeshStructure::Unstructured.
 */
enum class eMeshCellDimensionKind
{
  //! Les mailles ont la même dimension que le maillage
  MonoDimension,
  /*!
   * \brief Les mailles ont la même dimension que le maillage ou une dimension inférieure.
   *
   ** \warning Ce mode est expérimental.
   */
  MultiDimension,
  /*!
   * \brief Maillage non manifold.
   *
   * Le maillage est MultiDimension et non manifold.
   * Dans ce cas, si le maillage est 3D, les mailles 2D ont des arêtes (Edge)
   * au lieu des faces (Face).
   *
   * \warning Ce mode est expérimental.
   */
  NonManifold
};

extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& o, eMeshCellDimensionKind r);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Caractéristiques d'un maillage.
 *
 * Pour l'instant les caractéristiques sont :
 *
 * - la structure du maillage (eMeshStructure)
 * - le type d'AMR
 * - la gestion de la dimension des mailles (eMeshDimensionKind). Cette dernière
 *
 * \note Le support de maillages autres que eMeshDimensionKind.MonoDimension
 * est expérimental.
 */
class ARCANE_CORE_EXPORT MeshKind
{
 public:

  eMeshStructure meshStructure() const { return m_structure; }
  eMeshAMRKind meshAMRKind() const { return m_amr_kind; }
  eMeshCellDimensionKind meshDimensionKind() const { return m_dimension_kind; }
  //! Vrai si la structure du maillage est eMeshCellDimensionKind::NonManifold
  bool isNonManifold() const { return m_dimension_kind == eMeshCellDimensionKind::NonManifold; }
  //! Vrai si la structure du maillage est eMeshCellDimensionKind::MonoDimension
  bool isMonoDimension() const { return m_dimension_kind == eMeshCellDimensionKind::MonoDimension; }
  void setMeshStructure(eMeshStructure v) { m_structure = v; }
  void setMeshAMRKind(eMeshAMRKind v) { m_amr_kind = v; }
  void setMeshDimensionKind(eMeshCellDimensionKind v) { m_dimension_kind = v; }

 private:

  eMeshStructure m_structure = eMeshStructure::Unknown;
  eMeshAMRKind m_amr_kind = eMeshAMRKind::None;
  eMeshCellDimensionKind m_dimension_kind = eMeshCellDimensionKind::MonoDimension;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
