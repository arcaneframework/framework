// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshGenerator.cc                                   (C) 2000-2023 */
/*                                                                           */
/* Service de génération de maillage cartésien.                              */
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
  Real3 m_origine; // origine du maillage
  RealUniqueArray m_bloc_lx; // longueurs en x
  RealUniqueArray m_bloc_ly; // longueurs en y
  RealUniqueArray m_bloc_lz; // longueurs en z
  Int32UniqueArray m_bloc_nx; // nombre de mailles par bloc en x
  Int32UniqueArray m_bloc_ny; // nombre de mailles par bloc en y
  Int32UniqueArray m_bloc_nz; // nombre de mailles par bloc en z
  RealUniqueArray m_bloc_px; // progressions par bloc en x
  RealUniqueArray m_bloc_py; // progressions par bloc en y
  RealUniqueArray m_bloc_pz; // progressions par bloc en z
  Integer m_nsdx = 0; // nombre de sous-domaines en x
  Integer m_nsdy = 0; // nombre de sous-domaines en y
  Integer m_nsdz = 0; // nombre de sous-domaines en z
  //! Indique si on génère les groupes pour un cas test de sod
  bool m_is_generate_sod_groups = false;
  //! Version de l'algorithme de numérotation des faces
  Int32 m_face_numbering_version = -1;
  //! Version de l'algorithme de numérotation des arêtes
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
  RealUniqueArray m_bloc_ox; // origine bloc en x
  RealUniqueArray m_bloc_oy; // origine bloc longueurs en y
  RealUniqueArray m_bloc_oz; // origine bloc longueurs en z
  Real3 m_l; // longueurs en x, y et z
  Integer m_nx = 0; // nombre de mailles en x
  Integer m_ny = 0; // nombre de mailles en y
  Integer m_nz = 0; // nombre de mailles en z

 private:

  bool _readOptions();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
