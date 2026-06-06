// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshInternal.h                                             (C) 2000-2025 */
/*                                                                           */
/* Internal IMesh component in Arcane.                                       */
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
class IItemFamilySerializerMngInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Internal part of IMesh.
 */
class ARCANE_CORE_EXPORT IMeshInternal
{
 public:

  virtual ~IMeshInternal() = default;

 public:

  /*!
   * \brief Sets the mesh type.
   *
   * For now, this method should only be used to specify
   * the mesh structure (eMeshStructure).
   */
  virtual void setMeshKind(const MeshKind& v) = 0;

  /*!
   * \brief Returns the dof connectivity manager.
   *
   * This method is temporary because this dof connectivity manager
   * is intended to be removed, as the evolution of dof connectivities is now managed
   * automatically. For internal use only while awaiting removal.
   */
  virtual IItemConnectivityMng* dofConnectivityMng() const noexcept = 0;

  /*!
   * \bief Returns the polyhedral mesh modification interface
   *
   * This method returns nullptr if the mesh implementation is not PolyhedralMesh
   */
  virtual IPolyhedralMeshModifier* polyhedralMeshModifier() const noexcept = 0;

  /*!
   * \brief Returns the family serialization tools manager.
   *
   * This manager is used for the polyhedral mesh, in order to trigger a
   * finalization phase of item addition after calling the deserialization methods,
   * because these methods are asynchronous and the finalization phase must be triggered.
   *
   * @return This method returns nullptr if the manager does not exist.
   */
  virtual IItemFamilySerializerMngInternal* familySerializerMng() const noexcept { return nullptr; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
