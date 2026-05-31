// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshGenerator.cc                                   (C) 2000-2023 */
/*                                                                           */
/* Cartesian mesh generation service.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_CARTESIAN_MESHGENERATOR_H
#define ARCANE_STD_CARTESIAN_MESHGENERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/std/IMeshGenerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ICartesianMeshGenerationInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CartesianMeshGeneratorBuildInfo
{
 public:

  int m_mesh_dimension = -1;
  Real3 m_origine; // mesh origin
  RealUniqueArray m_bloc_lx; // lengths in x
  RealUniqueArray m_bloc_ly; // lengths in y
  RealUniqueArray m_bloc_lz; // lengths in z
  Int32UniqueArray m_bloc_nx; // number of meshes per block in x
  Int32UniqueArray m_bloc_ny; // number of meshes per block in y
  Int32UniqueArray m_bloc_nz; // number of meshes per block in z
  RealUniqueArray m_bloc_px; // progressions per block in x
  RealUniqueArray m_bloc_py; // progressions per block in y
  RealUniqueArray m_bloc_pz; // progressions per block in z
  Integer m_nsdx = 0; // number of sub-domains in x
  Integer m_nsdy = 0; // number of sub-domains in y
  Integer m_nsdz = 0; // number of sub-domains in z
  //! Indicates whether groups are generated for an SOD test case
  bool m_is_generate_sod_groups = false;
  //! Version of the face numbering algorithm
  Int32 m_face_numbering_version = -1;
  //! Version of the edge numbering algorithm
  Int32 m_edge_numbering_version = -1;

 public:

  void readOptionsFromXml(XmlNode cartesian_node);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CartesianMeshGenerator
: public TraceAccessor
, public IMeshGenerator
{
 public:

  explicit CartesianMeshGenerator(IPrimaryMesh* mesh);

 public:

  IntegerConstArrayView communicatingSubDomains() const override
  {
    return m_communicating_sub_domains;
  }
  bool readOptions(XmlNode node) override;
  bool generateMesh() override;
  void setBuildInfo(const CartesianMeshGeneratorBuildInfo& build_info);

 private:

  int sdXOffset();
  int sdYOffset();
  int sdZOffset();
  int sdXOffset(int);
  int sdYOffset(int);
  int sdZOffset(int);
  Int32 ownXNbCell();
  Int32 ownYNbCell();
  Int32 ownZNbCell();
  Int32 ownXNbCell(int);
  Int32 ownYNbCell(int);
  Int32 ownZNbCell(int);

 private:

  Real nxDelta(Real,int);
  Real nyDelta(Real,int);
  Real nzDelta(Real,int);
  Real xDelta(int);
  Real yDelta(int);
  Real zDelta(int);

 private:

  void xScan(const Int64,Int32Array&,Int32Array&,Int64Array&,
             Int64Array&,Int64Array&);
  void yScan(const Integer, Int32Array&,Int32Array&,Int64Array&,
             Int64Array&,Int64Array&,Int64,Int64);
  void zScan(const Int64, Int32Array&,Int32Array&,Int64Array&,
             Int64Array&,Int64Array&,Int64,Int64);

 private:

  IPrimaryMesh* m_mesh;
  Int32 m_my_mesh_part;
  UniqueArray<Int32> m_communicating_sub_domains;
  int m_mesh_dimension = -1;
  ICartesianMeshGenerationInfo* m_generation_info = nullptr;

 private:

  CartesianMeshGeneratorBuildInfo m_build_info;
  RealUniqueArray m_bloc_ox; // block origin in x
  RealUniqueArray m_bloc_oy; // block origin lengths in y
  RealUniqueArray m_bloc_oz; // block origin lengths in z
  Real3 m_l; // lengths in x, y, and z
  Integer m_nx = 0; // number of meshes in x
  Integer m_ny = 0; // number of meshes in y
  Integer m_nz = 0; // number of meshes in z

 private:

  bool _readOptions();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
