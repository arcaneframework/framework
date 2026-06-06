// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshUniqueIdMng.h                                          (C) 2000-2025 */
/*                                                                           */
/* Interface of the uniqueId() numbering manager of a mesh.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMESHUNIQUEIDMNG_H
#define ARCANE_IMESHUNIQUEIDMNG_H
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
 * \internal
 *
 * Interface of the uniqueId() numbering manager for mesh entities.
 *
 * This manager allows managing the calculation of uniqueIds() for entities
 * of the mesh that are implicitly created as faces or edges.
 */
class ARCANE_CORE_EXPORT IMeshUniqueIdMng
{
 public:

  //! Frees resources
  virtual ~IMeshUniqueIdMng() = default;

 public:

  /*!
   * \brief Sets the face numbering version.
   *
   * Valid values are 0, 1, 2, and 3. The default value is 1.
   * If the version is 0, there is no renumbering. In parallel,
   * the uniqueIds() of the faces must be consistent between
   * subdomains.
   */
  virtual void setFaceBuilderVersion(Integer n) = 0;

  //! Face numbering version.
  virtual Integer faceBuilderVersion() const = 0;

  /*!
   * \brief Sets the edge numbering version.
   *
   * Valid values are 0, 1, and 2. Value 1 works regardless of
   * the number of meshes, but the mesh must be read by
   * a single processor. Value 2 only works if the maximum of the
   * node uniqueIds() does not exceed 2^31.
   *
   * If the version is 0, there is no renumbering. In parallel,
   * the uniqueIds() of the faces must be consistent between
   * subdomains.
   */
  virtual void setEdgeBuilderVersion(Integer n) = 0;

  //! Edge numbering version.
  virtual Integer edgeBuilderVersion() const = 0;

  /*!
   * \brief Indicates whether the uniqueIds() of edges and faces
   * are determined based on the uniqueIds() of the nodes they consist of.
   *
   * This method must be called before setting the mesh dimension
   * (IPrimaryMesh::setDimension()).
   *
   * If active, when an edge or face is created on the fly,
   * MeshUtils::generateHashUniqueId() is used to generate the
   * uniqueId() of the entity. This allows automatically creating
   * edges or faces in parallel.
   *
   * \warning If this mechanism is used, it should not be mixed
   * with the manual creation of edges or faces (via IMeshModifier)
   * or you must use MeshUtils::generateHashUniqueId() to generate
   * the same identifier as the one created on the fly.
   */
  virtual void setUseNodeUniqueIdToGenerateEdgeAndFaceUniqueId(bool v) = 0;

  //! Indicates the mechanism used to number edges or faces
  virtual bool isUseNodeUniqueIdToGenerateEdgeAndFaceUniqueId() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
