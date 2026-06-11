// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICartesianMeshPatch.h                                       (C) 2000-2026 */
/*                                                                           */
/* Interface of an AMR patch of a Cartesian mesh.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_ICARTESIANMESHPATCH_H
#define ARCANE_CARTESIANMESH_ICARTESIANMESHPATCH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/AMRPatchPosition.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ICartesianMeshPatchInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Interface of an AMR patch of a Cartesian mesh.
 */
class ARCANE_CARTESIANMESH_EXPORT ICartesianMeshPatch
{
 public:

  virtual ~ICartesianMeshPatch() {} //<! Releases resources

  //! Cell group of the patch
  virtual CellGroup cells() = 0;
  virtual CellGroup inPatchCells() = 0;
  virtual CellGroup overlapCells() = 0;

  //! TODO
  virtual Integer index() = 0;

  //! List of cells in direction \a dir
  virtual CellDirectionMng& cellDirection(eMeshDirection dir) = 0;

  //! List of cells in direction \a dir (0, 1 or 2)
  virtual CellDirectionMng& cellDirection(Integer idir) = 0;

  //! List of faces in direction \a dir
  virtual FaceDirectionMng& faceDirection(eMeshDirection dir) = 0;

  //! List of faces in direction \a dir (0, 1 or 2)
  virtual FaceDirectionMng& faceDirection(Integer idir) = 0;

  //! List of nodes in direction \a dir
  virtual NodeDirectionMng& nodeDirection(eMeshDirection dir) = 0;

  //! List of nodes in direction \a dir (0, 1 or 2)
  virtual NodeDirectionMng& nodeDirection(Integer idir) = 0;

  //! Performs checks on the validity of the instance.
  virtual void checkValid() const = 0;

  virtual AMRPatchPosition position() const = 0;

  virtual ICartesianMeshPatchInternal* _internalApi() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
