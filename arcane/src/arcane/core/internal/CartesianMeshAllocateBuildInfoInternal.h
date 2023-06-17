// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAllocateBuildInfoInternal.h                    (C) 2000-2023 */
/*                                                                           */
/* Informations pour allouer les entités d'un maillage cartésien.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CARTESIANMESHALLOCATEBUILDINFOINTERNAL_H
#define ARCANE_CORE_CARTESIANMESHALLOCATEBUILDINFOINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CartesianMeshAllocateBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Partie interne de CartesianMeshAllocateBuildInfo.
 */
class ARCANE_CORE_EXPORT CartesianMeshAllocateBuildInfoInternal
{
  friend class CartesianMeshAllocateBuildInfo::Impl;

 public:

  Int32 meshDimension() const;

  //! Positionne la version utilisée pour le calcul des uniqueId() des faces
  void setFaceBuilderVersion(Int32 version);

  //! Version utilisée pour le calcul des des uniqueId() des faces
  Int32 faceBuilderVersion() const;

  //! Positionne la version utilisée pour le calcul des uniqueId() des arêtes
  void setEdgeBuilderVersion(Int32 version);

  //! Version utilisée pour le calcul des uniqueId() des arêtes
  Int32 edgeBuilderVersion() const;

 public:

  std::array<Int64, 3> globalNbCells() const;
  std::array<Int32, 3> ownNbCells() const;
  Int64 cellUniqueIdOffset() const;
  Int64 nodeUniqueIdOffset() const;

 private:

  CartesianMeshAllocateBuildInfo::Impl* m_p = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
