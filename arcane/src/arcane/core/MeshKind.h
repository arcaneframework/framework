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
  Unknown,
  Unstructured,
  Cartesian,
  Polyhedral
};

extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& o, eMeshStructure r);

//! Type de maillage AMR
enum class eMeshAMRKind
{
  None,
  Cell,
  Patch,
  PatchCartesianMeshOnly
};

extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& o, eMeshAMRKind r);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Caractéristiques d'un maillage.
 *
 * Pour l'instant les caractéristiques sont:
 * - la structure du maillage (eMeshStructure)
 * - le type d'AMR
 * - si le maillage est 'manifold' (le défaut) ou non.
 *
 * \note Le support de maillage non-manifold est expérimental.
 */
class ARCANE_CORE_EXPORT MeshKind
{
 public:

  eMeshStructure meshStructure() const { return m_structure; }
  eMeshAMRKind meshAMRKind() const { return m_amr_kind; }
  bool isNonManifold() const { return m_is_non_manifold; }

  void setMeshStructure(eMeshStructure v) { m_structure = v; }
  void setMeshAMRKind(eMeshAMRKind v) { m_amr_kind = v; }
  void setIsNonManifold(bool v) { m_is_non_manifold = v; }

 private:

  eMeshStructure m_structure = eMeshStructure::Unknown;
  eMeshAMRKind m_amr_kind = eMeshAMRKind::None;
  bool m_is_non_manifold = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
