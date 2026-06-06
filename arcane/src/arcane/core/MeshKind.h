// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshKind.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Characteristics of a mesh.                                                */
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

//! Mesh structure
enum class eMeshStructure
{
  //! Unknown or not initialized
  Unknown,
  //! Unstructured mesh
  Unstructured,
  //! Cartesian mesh
  Cartesian,
  //! Polyhedral mesh
  Polyhedral
};

extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& o, eMeshStructure r);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! AMR mesh type
enum class eMeshAMRKind
{
  //! The mesh is not AMR
  None,
  //! The mesh is AMR by cell
  Cell,
  //! The mesh is AMR by patch
  Patch,
  //! The mesh is AMR by Cartesian patch (rectangular)
  PatchCartesianMeshOnly
};

extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& o, eMeshAMRKind r);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Types of mesh dimension management.
 *
 * \warning Modes other than eMeshCellDimensionKind::MonoDimension are
 * experimental and are only supported for eMeshStructure::Unstructured.
 */
enum class eMeshCellDimensionKind
{
  //! The cells have the same dimension as the mesh
  MonoDimension,
  /*!
   * \brief The cells have the same dimension as the mesh or a lower dimension.
   *
   ** \warning This mode is experimental.
   */
  MultiDimension,
  /*!
   * \brief Non-manifold mesh.
   *
   * The mesh is MultiDimension and non-manifold.
   * In this case, if the mesh is 3D, the 2D cells have edges (Edge)
   * instead of faces (Face).
   *
   * \warning This mode is experimental.
   */
  NonManifold
};

extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& o, eMeshCellDimensionKind r);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Characteristics of a mesh.
 *
 * For now, the characteristics are:
 *
 * - the mesh structure (eMeshStructure)
 * - the AMR type
 * - the cell dimension management (eMeshDimensionKind). The latter
 *
 * \note Support for meshes other than eMeshDimensionKind.MonoDimension
 * is experimental.
 */
class ARCANE_CORE_EXPORT MeshKind
{
 public:

  eMeshStructure meshStructure() const { return m_structure; }
  eMeshAMRKind meshAMRKind() const { return m_amr_kind; }
  eMeshCellDimensionKind meshDimensionKind() const { return m_dimension_kind; }
  //! True if the mesh structure is eMeshCellDimensionKind::NonManifold
  bool isNonManifold() const { return m_dimension_kind == eMeshCellDimensionKind::NonManifold; }
  //! True if the mesh structure is eMeshCellDimensionKind::MonoDimension
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
