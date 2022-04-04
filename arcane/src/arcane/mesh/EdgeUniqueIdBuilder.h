﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EdgeUniqueIdBuilder.h                                       (C) 2000-2014 */
/*                                                                           */
/* Construction des indentifiants uniques des edges.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_EDGEUNIQUEIDBUILDER_H
#define ARCANE_MESH_EDGEUNIQUEIDBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SerializeBuffer;
class IParallelExchanger;
class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction des couches fantômes.
 */
class EdgeUniqueIdBuilder
: public TraceAccessor
{
 public:

  typedef DynamicMeshKindInfos::ItemInternalMap ItemInternalMap;
  typedef ItemInternalMap::Data ItemInternalMapData;

  typedef HashTableMapT<Int32,SharedArray<Int64> > BoundaryInfosMap;
  typedef HashTableMapEnumeratorT<Int32,SharedArray<Int64> > BoundaryInfosMapEnumerator;
  
 public:

  //! Construit une instance pour le maillage \a mesh
  EdgeUniqueIdBuilder(DynamicMeshIncrementalBuilder* mesh_builder);
  virtual ~EdgeUniqueIdBuilder();

 public:

  void computeEdgesUniqueIds();

 private:

  DynamicMesh* m_mesh;
  DynamicMeshIncrementalBuilder* m_mesh_builder;

 private:
  
  void _computeEdgesUniqueIdsSequential();
  void _computeEdgesUniqueIdsParallel3();
  void _exchangeData(IParallelExchanger* exchanger,BoundaryInfosMap& boundary_infos_to_send);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
