// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ARCANE_STD_INTERNAL_IOSGMSH_H
#define ARCANE_STD_INTERNAL_IOSGMSH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IMeshReader.h"

#define MSH_LIN_2 1 //1 2-node line.
#define MSH_TRI_3 2 //2 3-node triangle.
#define MSH_QUA_4 3 //3 4-node quadrangle.
#define MSH_TET_4 4 //4 4-node tetrahedron.
#define MSH_HEX_8 5 //5 8-node hexahedron.
#define MSH_PRI_6 6 //6 6-node prism.
#define MSH_PYR_5 7 //7 5-node pyramid.
#define MSH_LIN_3 8 //8 3-node second order line (2 nodes associated with the vertices and 1 with the edge).
#define MSH_TRI_6 9 //9 6-node second order triangle (3 nodes associated with the vertices and 3 with the edges).
#define MSH_QUA_9 10 //10 9-node second order quadrangle (4 nodes associated with the vertices, 4 with the edges and 1 with the face).
#define MSH_TET_10 11 //11 10-node second order tetrahedron (4 nodes associated with the vertices and 6 with the edges).
#define MSH_HEX_27 12 //12 27-node second order hexahedron (8 nodes associated with the vertices, 12 with the edges, 6 with the faces and 1 with the volume).
#define MSH_PRI_18 13 //13 18-node second order prism (6 nodes associated with the vertices, 9 with the edges and 3 with the quadrangular faces).
#define MSH_PYR_14 14 //14 14-node second order pyramid (5 nodes associated with the vertices, 8 with the edges and 1 with the quadrangular face).
#define MSH_PNT 15 //15 1-node point.
#define MSH_QUA_8 16 //16 8-node second order quadrangle (4 nodes associated with the vertices and 4 with the edges).
#define MSH_HEX_20 17 //17 20-node second order hexahedron (8 nodes associated with the vertices and 12 with the edges).
#define MSH_PRI_15 18 //18 15-node second order prism (6 nodes associated with the vertices and 9 with the edges).
#define MSH_PYR_13 19 //19 13-node second order pyramid (5 nodes associated with the vertices and 8 with the edges).
#define MSH_TRI_9 20
#define MSH_TRI_10 21
#define MSH_TRI_12 22
#define MSH_TRI_15 23
#define MSH_TRI_15I 24
#define MSH_TRI_21 25
#define MSH_LIN_4 26
#define MSH_LIN_5 27
#define MSH_LIN_6 28

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMshMeshReader
{
 public:

  virtual ~IMshMeshReader() = default;

 public:

  virtual IMeshReader::eReturnType
  readMeshFromMshFile(IMesh* mesh, const String& file_name, bool use_internal_partition) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
