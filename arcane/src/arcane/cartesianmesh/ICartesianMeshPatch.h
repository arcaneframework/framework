// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICartesianMeshPatch.h                                       (C) 2000-2025 */
/*                                                                           */
/* Interface d'un patch AMR d'un maillage cartésien.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_ICARTESIANMESHPATCH_H
#define ARCANE_CARTESIANMESH_ICARTESIANMESHPATCH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/AMRPatchPosition.h"
#include "arcane/ItemTypes.h"
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
 * \brief Interface d'un patch AMR d'un maillage cartésien.
 */
class ARCANE_CARTESIANMESH_EXPORT ICartesianMeshPatch
{
 public:

  virtual ~ICartesianMeshPatch() {} //<! Libère les ressources

  //! Groupe de mailles du patch
  virtual CellGroup cells() =0;
  virtual CellGroup inPatchCells() = 0;

  //! TODO
  virtual Integer index() = 0;

  //! Liste des mailles dans la direction \a dir
  virtual CellDirectionMng& cellDirection(eMeshDirection dir) =0;

  //! Liste des mailles dans la direction \a dir (0, 1 ou 2)
  virtual CellDirectionMng& cellDirection(Integer idir) =0;

  //! Liste des faces dans la direction \a dir
  virtual FaceDirectionMng& faceDirection(eMeshDirection dir) =0;

  //! Liste des faces dans la direction \a dir (0, 1 ou 2)
  virtual FaceDirectionMng& faceDirection(Integer idir) =0;

  //! Liste des noeuds dans la direction \a dir
  virtual NodeDirectionMng& nodeDirection(eMeshDirection dir) =0;

  //! Liste des noeuds dans la direction \a dir (0, 1 ou 2)
  virtual NodeDirectionMng& nodeDirection(Integer idir) =0;

  //! Effectue des vérifications sur la validité de l'instance.
  virtual void checkValid() const =0;

  virtual AMRPatchPosition position() const = 0;

  virtual ICartesianMeshPatchInternal* _internalApi() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

