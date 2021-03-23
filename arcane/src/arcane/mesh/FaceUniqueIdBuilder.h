// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceUniqueIdBuilder.h                                       (C) 2000-2020 */
/*                                                                           */
/* Construction des indentifiants uniques des faces.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_FACEUNIQUEIDBUILDER_H
#define ARCANE_MESH_FACEUNIQUEIDBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction des couches fantômes.
 */
class FaceUniqueIdBuilder
: public TraceAccessor
{
 public:

  typedef DynamicMeshKindInfos::ItemInternalMap ItemInternalMap;
  typedef ItemInternalMap::Data ItemInternalMapData;

  typedef HashTableMapT<Int32,SharedArray<Int64> > BoundaryInfosMap;
  typedef HashTableMapEnumeratorT<Int32,SharedArray<Int64> > BoundaryInfosMapEnumerator;
  
 public:

  //! Construit une instance pour le maillage \a mesh
  explicit FaceUniqueIdBuilder(DynamicMeshIncrementalBuilder* mesh_builder);
  virtual ~FaceUniqueIdBuilder();

 public:

  void computeFacesUniqueIds();

 private:

  DynamicMesh* m_mesh;
  DynamicMeshIncrementalBuilder* m_mesh_builder;

 private:
  
  void _computeFacesUniqueIdsSequential();
  void _computeFacesUniqueIdsParallel2();
  void _computeFacesUniqueIdsParallel3();
  void _computeFacesUniqueIdsFast();
  void _exchangeData(IParallelExchanger* exchanger,BoundaryInfosMap& boundary_infos_to_send);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
