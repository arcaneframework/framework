// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshGenerator.cc                                   (C) 2000-2025 */
/*                                                                           */
/* Service de génération de maillage cartésien.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/CartesianMeshGenerator.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/Vector3.h"
#include "arcane/utils/Vector2.h"

#include "arcane/core/IMeshReader.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ICaseDocument.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/XmlNodeIterator.h"
#include "arcane/core/Service.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/Item.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/MeshVariable.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/AbstractService.h"
#include "arcane/core/Properties.h"
#include "arcane/core/MeshPartInfo.h"
#include "arcane/core/IMeshBuilder.h"
#include "arcane/core/IMeshUniqueIdMng.h"
#include "arcane/core/IMeshInitialAllocator.h"
#include "arcane/core/ICartesianMeshGenerationInfo.h"
#include "arcane/core/CartesianMeshAllocateBuildInfo.h"
#include "arcane/core/internal/CartesianMeshAllocateBuildInfoInternal.h"

#include "arcane/std/Cartesian2DMeshGenerator_axl.h"
#include "arcane/std/Cartesian3DMeshGenerator_axl.h"

#include "arcane/std/internal/SodStandardGroupsBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshGeneratorBuildInfo::
readOptionsFromXml(XmlNode cartesian_node)
{
  XmlNode origine_node = cartesian_node.child("origine");
  XmlNode nsd_node = cartesian_node.child("nsd");
  XmlNodeList lx_node_list = cartesian_node.children("lx");
  XmlNodeList ly_node_list = cartesian_node.children("ly");
  XmlNodeList lz_node_list = cartesian_node.children("lz");

  // On vérifie qu'on a bien au moins trois données x, y & z et une origine
  if (origine_node.null())
    ARCANE_FATAL("No <origin> element found");
  if (lx_node_list.size() == 0)
    ARCANE_FATAL("No <lx> elements found");
  if (ly_node_list.size() == 0)
    ARCANE_FATAL("No <ly> elements found");

  String origine_value = origine_node.value();
  // On récupère l'origine
  bool is_bad_origin = builtInGetValue(m_origine, origine_value);

  // On regarde si on a un noeud en Z
  if (lz_node_list.size() == 0) {
    m_mesh_dimension = 2;
    // En 2D, on autorise l'origine a avoir 2 composantes
    if (is_bad_origin) {
      Real2 xy(0.0, 0.0);
      is_bad_origin = builtInGetValue(xy, origine_value);
      if (!is_bad_origin) {
        m_origine.x = xy.x;
        m_origine.y = xy.y;
      }
    }
  }
  else {
    m_mesh_dimension = 3;
  }
  if (is_bad_origin)
    ARCANE_FATAL("Element '{0}' : can not convert value '{1}' to type Real3", origine_node.xpathFullName(), origine_value);

  // On récupère les longueurs des blocs, + true pour throw_exception
  // On récupère aussi les nombres de mailles des blocs + true pour throw_exception
  // On récupère aussi les progressions géométriques
  // On met les progressions à 1.0 par défaut
  for (XmlNode& lx_node : lx_node_list) {
    m_bloc_lx.add(lx_node.valueAsReal(true));
    m_bloc_nx.add(lx_node.attr("nx", true).valueAsInteger(true));
    Real px = lx_node.attr("prx").valueAsReal(true);
    if (px == 0.0)
      px = 1.0;
    m_bloc_px.add(px);
  }
  for (XmlNode& ly_node : ly_node_list) {
    m_bloc_ly.add(ly_node.valueAsReal(true));
    m_bloc_ny.add(ly_node.attr("ny", true).valueAsInteger(true));
    Real py = ly_node.attr("pry").valueAsReal(true);
    if (py == 0.0)
      py = 1.0;
    m_bloc_py.add(py);
  }
  if (m_mesh_dimension == 3) {
    for (XmlNode& lz_node : lz_node_list) {
      m_bloc_lz.add(lz_node.valueAsReal(true));
      m_bloc_nz.add(lz_node.attr("nz", true).valueAsInteger(true));
      Real pz = lz_node.attr("prz").valueAsReal(true);
      if (pz == 0.0)
        pz = 1.0;
      m_bloc_pz.add(pz);
    }
  }

  // On récupère les nombres de sous-domaines + throw_exception
  String nsd_value = nsd_node.value();
  IntegerUniqueArray nsd;
  if (builtInGetValue(nsd, nsd_value))
    ARCANE_FATAL("Can not convert string '{0}' to Int[]", nsd_value);
  if (nsd.size() != m_mesh_dimension)
    ARCANE_FATAL("Number of sub-domain '<nsd>={0}' has to be equal to mesh dimension '{1}'",
                 nsd.size(), m_mesh_dimension);

  m_nsdx = nsd[0];
  m_nsdy = nsd[1];
  m_nsdz = (m_mesh_dimension == 3) ? nsd[2] : 0;

  {
    XmlNode version_node = cartesian_node.child("face-numbering-version");
    if (!version_node.null()){
      Int32 v = version_node.valueAsInteger(true);
      if (v>=0)
        m_face_numbering_version = v;
    }
  }
  {
    XmlNode version_node = cartesian_node.child("edge-numbering-version");
    if (!version_node.null()){
      Int32 v = version_node.valueAsInteger(true);
      if (v>=0)
        m_edge_numbering_version = v;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshGenerator::
CartesianMeshGenerator(IPrimaryMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_my_mesh_part(mesh->parallelMng()->commRank())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CartesianMeshGenerator::
readOptions(XmlNode cartesian_node)
{
  m_build_info.readOptionsFromXml(cartesian_node);
  return _readOptions();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshGenerator::
setBuildInfo(const CartesianMeshGeneratorBuildInfo& build_info)
{
  m_build_info = build_info;
  _readOptions();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief readOptions
 */
bool CartesianMeshGenerator::
_readOptions()
{
  Trace::Setter mci(traceMng(),"CartesianMeshGenerator");

  m_mesh_dimension = m_build_info.m_mesh_dimension;
  Int32 nb_sub_domain = m_mesh->parallelMng()->commSize();

  // On met les nombres de sous-domaines par défaut
  if (m_build_info.m_nsdx == 0 || nb_sub_domain == 1)
    m_build_info.m_nsdx = 1;
  if (m_build_info.m_nsdy == 0 || nb_sub_domain == 1)
    m_build_info.m_nsdy = 1;
  if (m_build_info.m_nsdz == 0 || nb_sub_domain == 1)
    m_build_info.m_nsdz = 1;

  // Synthèse des longueurs des blocs
  m_l.x = m_l.y = m_l.z = 0.;
  for (int i = 0; i < m_build_info.m_bloc_lx.size(); i += 1)
    m_l.x += m_build_info.m_bloc_lx.at(i);
  for (int i = 0; i < m_build_info.m_bloc_ly.size(); i += 1)
    m_l.y += m_build_info.m_bloc_ly.at(i);
  if (m_mesh_dimension == 3)
    for (int i = 0; i < m_build_info.m_bloc_lz.size(); i += 1)
      m_l.z += m_build_info.m_bloc_lz.at(i);

  // Synthèse des origines pour chaque bloc
  m_bloc_ox.add(m_build_info.m_origine.x);
  m_bloc_oy.add(m_build_info.m_origine.y);
  if (m_mesh_dimension == 3)
    m_bloc_oz.add(m_build_info.m_origine.z);
  for (int i = 0; i < m_build_info.m_bloc_lx.size(); i += 1)
    m_bloc_ox.add(m_bloc_ox.at(i) + m_build_info.m_bloc_lx.at(i));
  for (int i = 0; i < m_build_info.m_bloc_ly.size(); i += 1)
    m_bloc_oy.add(m_bloc_oy.at(i) + m_build_info.m_bloc_ly.at(i));
  if (m_mesh_dimension == 3)
    for (int i = 0; i < m_build_info.m_bloc_lz.size(); i += 1)
      m_bloc_oz.add(m_bloc_oz.at(i) + m_build_info.m_bloc_lz.at(i));

  // Synthèse des nombres de mailles des blocs et en total
  m_nx = m_ny = m_nz = 0;

  for (int i = 0; i < m_build_info.m_bloc_nx.size(); ++i)
    m_nx += m_build_info.m_bloc_nx.at(i);
  for (int i = 0; i < m_build_info.m_bloc_ny.size(); ++i)
    m_ny += m_build_info.m_bloc_ny.at(i);

  if (m_mesh_dimension == 3)
    for (int i = 0; i < m_build_info.m_bloc_nz.size(); i += 1)
      m_nz += m_build_info.m_bloc_nz.at(i);
  else
    m_nz += 1;

  // On dump les infos récupérées jusque là
  info() << " mesh_name=" << m_mesh->name();
  info() << " dimension=" << m_mesh_dimension;
  info() << " origin:" << m_build_info.m_origine;
  info() << " length x =" << m_l.x << "," << m_build_info.m_bloc_lx;
  info() << " length y="  << m_l.y << "," << m_build_info.m_bloc_ly;
  if (m_build_info.m_mesh_dimension == 3)
    info() << " length z=" << m_l.z << "," << m_build_info.m_bloc_lz;
  info() << "cells number:" << m_nx << "x" << m_ny << "x" << m_nz;
  info() << "progression x:" << m_build_info.m_bloc_px;
  info() << "progression y:" << m_build_info.m_bloc_py;
  if (m_mesh_dimension == 3)
    info() << "progression z:" << m_build_info.m_bloc_pz;
  info() << " nb_sub_domain:" << nb_sub_domain;
  info() << " decomposing the subdomains:" << m_build_info.m_nsdx << "x"
         << m_build_info.m_nsdy << "x" << m_build_info.m_nsdz;

  // Vérification du nombre de sous domaines vs ce qui a été spécifié
  if (m_build_info.m_nsdx * m_build_info.m_nsdy * m_build_info.m_nsdz != nb_sub_domain)
    ARCANE_FATAL("Specified partition {0}x{1}x{2} has to be equal to number of parts ({3})",
                 m_build_info.m_nsdx,m_build_info.m_nsdy,m_build_info.m_nsdz,nb_sub_domain);

  return false; // false == ok
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// ******************************************************************************
// * sd[X|Y|Z]Offset
// ******************************************************************************
inline Integer CartesianMeshGenerator::
sdXOffset(Int32 this_sub_domain_id)
{
  return this_sub_domain_id % m_build_info.m_nsdx;
}

inline Integer CartesianMeshGenerator::
sdYOffset(Int32 this_sub_domain_id)
{
  return (this_sub_domain_id / m_build_info.m_nsdx) % m_build_info.m_nsdy;
}

inline Integer CartesianMeshGenerator::
sdZOffset(Int32 this_sub_domain_id)
{
  return (this_sub_domain_id / (m_build_info.m_nsdx * m_build_info.m_nsdy)) % m_build_info.m_nsdz;
}

inline Integer CartesianMeshGenerator::
sdXOffset()
{
  return sdXOffset(m_my_mesh_part);
}

inline Integer CartesianMeshGenerator::
sdYOffset()
{
  return sdYOffset(m_my_mesh_part);
}

inline Integer CartesianMeshGenerator::
sdZOffset()
{
  return sdZOffset(m_my_mesh_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// ******************************************************************************
// * own[X|Y|Z]NbCell
// ******************************************************************************
inline Int32 ownNbCell(Int64 n, Integer nsd, int sd_offset)
{
  Int64 q = n / nsd;
  Int64 r = n % nsd;
  // Si on est en 'première approche', on a pas de reste
  if (r == 0)
    return CheckedConvert::toInt32(q);
  // Sinon, le reste des mailles est réparti sur les derniers sous domaines
  if (sd_offset < (nsd - r))
    return CheckedConvert::toInt32(q);
  return CheckedConvert::toInt32(q + 1);
}

inline Int32 CartesianMeshGenerator::
ownXNbCell()
{
  return ownNbCell(m_nx, m_build_info.m_nsdx, sdXOffset());
}
inline Int32 CartesianMeshGenerator::
ownXNbCell(int isd)
{
  return ownNbCell(m_nx, m_build_info.m_nsdx, sdXOffset(isd));
}
inline Int32 CartesianMeshGenerator::
ownYNbCell()
{
  return ownNbCell(m_ny, m_build_info.m_nsdy, sdYOffset());
}
inline Int32 CartesianMeshGenerator::
ownYNbCell(int isd)
{
  return ownNbCell(m_ny, m_build_info.m_nsdy, sdYOffset(isd));
}
inline Int32 CartesianMeshGenerator::
ownZNbCell()
{
  return ownNbCell(m_nz, m_build_info.m_nsdz, sdZOffset());
}
inline Int32 CartesianMeshGenerator::
ownZNbCell(int isd)
{
  return ownNbCell(m_nz, m_build_info.m_nsdz, sdZOffset(isd));
}

// ******************************************************************************
// * iProgression
// ******************************************************************************
inline Real
iDelta(int iBloc,
       const RealArray& bloc_p,
       const Int32Array& bloc_n,
       const RealArray& bloc_l,
       const Real all_l,
       const Int64 all_n)
{
  Real p = bloc_p.at(iBloc);
  Real n = (Real)bloc_n.at(iBloc);
  Real l = bloc_l.at(iBloc);
  if (p <= 1.0)
    return all_l / (Real)all_n;
  return (p - 1.0) / (math::pow(p, n) - 1.0) * l;
}

inline Real
kDelta(Real k,
       int iBloc,
       const RealArray& bloc_p,
       const Int32Array& bloc_n,
       const RealArray& bloc_l,
       const Real all_l,
       const Int64 all_n)
{
  ARCANE_UNUSED(all_l);
  ARCANE_UNUSED(all_n);

  Real p = bloc_p.at(iBloc);
  Real n = (Real)bloc_n.at(iBloc);
  Real l = bloc_l.at(iBloc);
  if (p == 1.0)
    return l * k / n;
  return l * (math::pow(p, k) - 1.0) / (math::pow(p, (Real)n) - 1.0);
}

inline Real CartesianMeshGenerator::
nxDelta(Real k, int iBloc)
{
  return m_bloc_ox.at(iBloc) + kDelta(k, iBloc, m_build_info.m_bloc_px, m_build_info.m_bloc_nx, m_build_info.m_bloc_lx, m_l.x, m_nx);
}
inline Real CartesianMeshGenerator::
xDelta(int iBloc)
{
  return iDelta(iBloc, m_build_info.m_bloc_px, m_build_info.m_bloc_nx, m_build_info.m_bloc_lx, m_l.x, m_nx);
}

inline Real CartesianMeshGenerator::
nyDelta(Real k, int iBloc)
{
  return m_bloc_oy.at(iBloc) + kDelta(k, iBloc, m_build_info.m_bloc_py, m_build_info.m_bloc_ny, m_build_info.m_bloc_ly, m_l.y, m_ny);
}
inline Real CartesianMeshGenerator::
yDelta(int iBloc)
{
  return iDelta(iBloc, m_build_info.m_bloc_py, m_build_info.m_bloc_ny, m_build_info.m_bloc_ly, m_l.y, m_ny);
}

inline Real CartesianMeshGenerator::
nzDelta(Real k, int iBloc)
{
  return m_bloc_oz.at(iBloc) + kDelta(k, iBloc, m_build_info.m_bloc_pz, m_build_info.m_bloc_nz, m_build_info.m_bloc_lz, m_l.z, m_nz);
}

inline Real CartesianMeshGenerator::
zDelta(int iBloc)
{
  return iDelta(iBloc, m_build_info.m_bloc_pz, m_build_info.m_bloc_nz, m_build_info.m_bloc_lz, m_l.z, m_nz);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// ******************************************************************************
// * X scanning
// ******************************************************************************
void CartesianMeshGenerator::
xScan(const Int64 all_nb_cell_x,
      IntegerArray& sd_x_ibl,
      IntegerArray& sd_x_obl,
      Int64Array& sd_x_nbl,
      Int64Array& sd_x_node_offset,
      Int64Array& sd_x_cell_offset)
{
  sd_x_ibl.add(0);
  sd_x_obl.add(0);
  sd_x_nbl.add(0);
  sd_x_node_offset.add(0);
  sd_x_cell_offset.add(0);
  // Reset des indices, offset et frontières
  Int64 nsd = 0;
  Integer isd = 0;
  Integer ibl = 0;
  Integer obl = 0;
  Int64 nbl = 0;
  for (Int64 x = 0; x < all_nb_cell_x; ++x) {
    // Si on découvre la frontière d'un bloc
    if (x == (nbl + m_build_info.m_bloc_nx.at(ibl))) {
      nbl += m_build_info.m_bloc_nx.at(ibl);
      info(4) << "\t[2;33m[CartesianMeshGenerator::xScan] Scan hit x bloc boundary: @ node " << nbl << "[0m";
      // On saute de bloc
      ibl += 1;
      // On reset l'offset de calcul dans le bloc
      obl = 0;
    }
    // Si on découvre la frontière d'un sous-domaine
    if (x == (nsd + ownXNbCell(isd))) {
      nsd += ownXNbCell(isd);
      info(4) << "\t[2;33m[CartesianMeshGenerator::xScan] Scan hit x sub domain boundary: @ node " << nsd << "[0m";
      // On sauvegarde les infos nécessaire à la reprise par sous-domaine
      sd_x_ibl.add(ibl);
      sd_x_obl.add(obl);
      sd_x_nbl.add(nbl);
      sd_x_node_offset.add(nsd);
      sd_x_cell_offset.add(nsd);
      isd += 1;
      info(4) << "\t[2;33m[CartesianMeshGenerator::xScan] Saving state: "
              << ", node_offset=" << sd_x_node_offset.at(isd)
              << ", cell_offset=" << sd_x_cell_offset.at(isd)
              << ", ibl=" << sd_x_ibl.at(isd)
              << ", nbl=" << sd_x_nbl.at(isd) << "[0m";
    }
    // On incrémente l'offset de calcul dans le bloc
    obl += 1;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// ******************************************************************************
// * Y scanning
// ******************************************************************************
void CartesianMeshGenerator::
yScan(const Integer all_nb_cell_y,
      IntegerArray& sd_y_ibl,
      IntegerArray& sd_y_obl,
      Int64Array& sd_y_nbl,
      Int64Array& sd_y_node_offset,
      Int64Array& sd_y_cell_offset,
      Int64 all_nb_node_x,
      Int64 all_nb_cell_x)
{
  sd_y_ibl.add(0);
  sd_y_obl.add(0);
  sd_y_nbl.add(0);
  sd_y_node_offset.add(0);
  sd_y_cell_offset.add(0);
  Int64 nsd = 0;
  Integer isd = 0;
  Integer ibl = 0;
  Integer obl = 0;
  Int64 nbl = 0;
  for (Int64 y = 0; y < all_nb_cell_y; ++y) {
    if (y == (nbl + m_build_info.m_bloc_ny.at(ibl))) {
      nbl += m_build_info.m_bloc_ny.at(ibl);
      info(4) << "\t[2;33m[CartesianMeshGenerator::generateMesh] Scan hit y bloc boundary: @ node " << nbl << "[0m";
      ibl += 1;
      obl = 0;
    }
    if (y == (nsd + ownYNbCell(isd))) {
      nsd += ownYNbCell(isd);
      info(4) << "\t[2;33m[CartesianMeshGenerator::generateMesh] Scan hit y sub domain boundary: @ node " << nsd << "[0m";
      sd_y_ibl.add(ibl);
      sd_y_obl.add(obl);
      sd_y_nbl.add(nbl * all_nb_node_x);
      sd_y_node_offset.add(nsd * all_nb_node_x);
      sd_y_cell_offset.add(nsd * all_nb_cell_x);
      isd += m_build_info.m_nsdx;
    }
    obl += 1;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// ******************************************************************************
// * Z scanning
// ******************************************************************************
void CartesianMeshGenerator::
zScan(const Int64 all_nb_cell_z,
      IntegerArray& sd_z_ibl,
      IntegerArray& sd_z_obl,
      Int64Array& sd_z_nbl,
      Int64Array& sd_z_node_offset,
      Int64Array& sd_z_cell_offset,
      Int64 all_nb_node_xy,
      Int64 all_nb_cell_xy)
{
  if (m_mesh_dimension != 3)
    return;
  sd_z_ibl.add(0);
  sd_z_obl.add(0);
  sd_z_nbl.add(0);
  sd_z_node_offset.add(0);
  sd_z_cell_offset.add(0);
  Int64 nsd = 0;
  Integer isd = 0;
  Integer ibl = 0;
  Integer obl = 0;
  Int64 nbl = 0;
  for (Int64 z = 0; z < all_nb_cell_z; ++z) {
    if (z == (nbl + m_build_info.m_bloc_nz.at(ibl))) {
      nbl += m_build_info.m_bloc_nz.at(ibl);
      ibl += 1;
      obl = 0;
    }
    if (z == (nsd + ownZNbCell(isd))) {
      nsd += ownZNbCell(isd);
      sd_z_ibl.add(ibl);
      sd_z_obl.add(obl);
      sd_z_nbl.add(nbl * all_nb_node_xy);
      sd_z_node_offset.add(nsd * all_nb_node_xy);
      sd_z_cell_offset.add(nsd * all_nb_cell_xy);
      isd += m_build_info.m_nsdx * m_build_info.m_nsdy;
    }
    obl += 1;
  }
}

// ******************************************************************************
// * generateMesh
// ******************************************************************************

bool CartesianMeshGenerator::
generateMesh()
{
  Trace::Setter mci(traceMng(),"CartesianMeshGenerator");
  IPrimaryMesh* mesh = m_mesh;

  m_generation_info = ICartesianMeshGenerationInfo::getReference(mesh,true);

  CartesianMeshAllocateBuildInfo cartesian_mesh_build_info(mesh);

  info() << " decomposing the subdomains:" << m_build_info.m_nsdx << "x"
         << m_build_info.m_nsdy << "x" << m_build_info.m_nsdz;
  info() << "sub domain offset @ " << sdXOffset() << "x" << sdYOffset() << "x" << sdZOffset();
  // All Cells Setup
  Integer all_nb_cell_x = m_nx;
  Integer all_nb_cell_y = m_ny;
  Integer all_nb_cell_z = m_nz;
  // Positionne des propriétés sur le maillage pour qu'il puisse connaître
  // le nombre de mailles dans chaque direction ainsi que l'offset du sous-domaine.
  // Cela est utilisé notammement par CartesianMesh.
  //Properties* mesh_properties = mesh->properties();
  m_generation_info->setGlobalNbCells(all_nb_cell_x,all_nb_cell_y,all_nb_cell_z);
  m_generation_info->setSubDomainOffsets(sdXOffset(),sdYOffset(),sdZOffset());
  m_generation_info->setNbSubDomains(m_build_info.m_nsdx,m_build_info.m_nsdy,m_build_info.m_nsdz);

  m_generation_info->setGlobalOrigin(m_build_info.m_origine);
  m_generation_info->setGlobalLength(m_l);

  Int64 all_nb_cell_xy = ((Int64)all_nb_cell_x) * ((Int64)all_nb_cell_y);
  Int64 all_nb_cell_xyz = ((Int64)all_nb_cell_xy) * ((Int64)all_nb_cell_z);
  info() << " all cells: " << all_nb_cell_x << "x" << all_nb_cell_y << "y"
         << all_nb_cell_z << "=" << all_nb_cell_xyz;

  // Own Cells Setup
  Int32 own_nb_cell_x = ownXNbCell();
  Int32 own_nb_cell_y = ownYNbCell();
  Int32 own_nb_cell_z = ownZNbCell();
  m_generation_info->setOwnNbCells(own_nb_cell_x,own_nb_cell_y,own_nb_cell_z);
  Integer own_nb_cell_xy = CheckedConvert::multiply(own_nb_cell_x, own_nb_cell_y);
  Integer own_nb_cell_xyz = CheckedConvert::multiply(own_nb_cell_xy, own_nb_cell_z);
  info() << " own cells: " << own_nb_cell_x << "x" << own_nb_cell_y << "y"
         << own_nb_cell_z << "=" << own_nb_cell_xyz;

  // All Nodes Setup
  Integer all_nb_node_x = all_nb_cell_x + 1;
  Integer all_nb_node_y = all_nb_cell_y + 1;
  Int64 all_nb_node_xy = ((Int64)all_nb_node_x) * ((Int64)all_nb_node_y);

  // Own Nodes Setup
  Integer own_nb_node_x = own_nb_cell_x + 1;
  Integer own_nb_node_y = own_nb_cell_y + 1;
  Integer own_nb_node_z = (m_mesh_dimension == 3) ? own_nb_cell_z + 1 : own_nb_cell_z;
  Integer own_nb_node_xy = CheckedConvert::multiply(own_nb_node_x, own_nb_node_y);
  Integer own_nb_node_xyz = CheckedConvert::multiply(own_nb_node_xy, own_nb_node_z);
  info() << " own nodes: "
         << own_nb_node_x << "x"
         << own_nb_node_y << "y"
         << own_nb_node_z << "=" << own_nb_node_xyz;

  // Création hash-table des noeuds et préparation de leurs uid
  UniqueArray<Int64> nodes_unique_id(CheckedConvert::toInteger(own_nb_node_xyz));
  HashTableMapT<Int64, NodeInfo> nodes_infos(CheckedConvert::toInteger(own_nb_node_xyz), true);

  // Calcule des infos blocs et des offset associés à tous les sub_domain_id
  IntegerUniqueArray sd_x_ibl, sd_x_obl; // Numéro et offset dans le bloc
  IntegerUniqueArray sd_y_ibl, sd_y_obl;
  IntegerUniqueArray sd_z_ibl, sd_z_obl;
  Int64UniqueArray sd_x_nbl, sd_y_nbl, sd_z_nbl; // Nombre dans le bloc
  Int64UniqueArray sd_x_node_offset;
  Int64UniqueArray sd_y_node_offset;
  Int64UniqueArray sd_z_node_offset;
  Int64UniqueArray sd_x_cell_offset;
  Int64UniqueArray sd_y_cell_offset;
  Int64UniqueArray sd_z_cell_offset;

  // Scan des sous domaines: i_ème (isd) et n_ème (nsd) noeud pour marquer la frontière sous domaine
  // Scan des des blocs:     i_ème (ibl), o_ème offset (obl) et n_ème (nbl) noeud pour marquer la frontière bloc

  // Selon la direction X
  xScan(all_nb_cell_x,
        sd_x_ibl, sd_x_obl, sd_x_nbl,
        sd_x_node_offset, sd_x_cell_offset);

  // Selon la direction Y
  yScan(all_nb_cell_y,
        sd_y_ibl, sd_y_obl, sd_y_nbl,
        sd_y_node_offset, sd_y_cell_offset,
        all_nb_node_x, all_nb_cell_x);

  // Selon la direction Z
  zScan(all_nb_cell_z,
        sd_z_ibl, sd_z_obl, sd_z_nbl,
        sd_z_node_offset, sd_z_cell_offset,
        all_nb_node_xy, all_nb_cell_xy);

  // Calcule pour les mailles l'offset global du début de la grille pour chaque direction.
  Int64x3 first_own_cell_offset;
  {
    Int64 cell_offset_x = sd_x_cell_offset[sdXOffset()];
    Int64 cell_offset_y = sd_y_cell_offset[sdYOffset()] / all_nb_cell_x;
    Int64 cell_offset_z = 0;
    if (m_mesh_dimension == 3) {
      cell_offset_z = sd_z_cell_offset[sdZOffset()] / all_nb_cell_xy;
    }
    first_own_cell_offset = Int64x3(cell_offset_x,cell_offset_y,cell_offset_z);
    info() << "OwnCellOffset info X=" << cell_offset_x << " Y=" << cell_offset_y << " Z=" << cell_offset_z;
    m_generation_info->setOwnCellOffsets(cell_offset_x,cell_offset_y,cell_offset_z);
  }
  // IBL, NBL
  info() << " sd_x_ibl=" << sd_x_ibl;
  info() << " sd_x_obl=" << sd_x_obl;
  info() << " sd_x_nbl=" << sd_x_nbl;

  info() << " sd_y_ibl=" << sd_y_ibl;
  info() << " sd_y_obl=" << sd_y_obl;
  info() << " sd_y_nbl=" << sd_y_nbl;

  info() << " sd_z_ibl=" << sd_z_ibl;
  info() << " sd_z_obl=" << sd_z_obl;
  info() << " sd_z_nbl=" << sd_z_nbl;

  // NODE OFFSET
  info() << " sd_x_node_offset=" << sd_x_node_offset;
  info() << " sd_y_node_offset=" << sd_y_node_offset;
  info() << " sd_z_node_offset=" << sd_z_node_offset;

  // CELL OFFSET
  info() << " sd_x_cell_offset=" << sd_x_cell_offset;
  info() << " sd_y_cell_offset=" << sd_y_cell_offset;
  info() << " sd_z_cell_offset=" << sd_z_cell_offset;

  // On calcule le premier node_local_id en fonction de notre sub_domain_id
  Integer node_local_id = 0;
  Int64 node_unique_id_offset = 0;
  info() << " sdXOffset=" << sdXOffset()
         << " sdYOffset=" << sdYOffset()
         << " sdZOffset=" << sdZOffset();
  // On calcule l'offset en x des node_unique_id
  node_unique_id_offset += sd_x_node_offset.at(sdXOffset());
  info() << " node_unique_id_offset=" << node_unique_id_offset;
  // On calcule l'offset en y des node_unique_id
  node_unique_id_offset += (sd_y_node_offset.at(sdYOffset()));
  info() << " node_unique_id_offset=" << node_unique_id_offset;
  if (m_mesh_dimension == 3) {
    // On calcule l'offset en z des node_unique_id
    node_unique_id_offset += (sd_z_node_offset.at(sdZOffset()));
    info() << " node_unique_id_offset=" << node_unique_id_offset;
  }

  Integer z_ibl = 0;
  Integer z_obl = 0;
  Int64 z_nbl = 0;
  if (m_mesh_dimension == 3) {
    z_ibl = sd_z_ibl.at(sdZOffset());
    z_obl = sd_z_obl.at(sdZOffset());
    z_nbl = sd_z_nbl.at(sdZOffset());
  }
  for (Int64 z = 0; z < own_nb_node_z; ++z) {
    if (m_mesh_dimension == 3) {
      // On saute l'éventuelle dernière frontière bl coincidante avec les sd
      // Et si on découvre une frontière bloc, on reset les origines
      if (((z + 1) != own_nb_node_z) &&
          ((z * all_nb_node_xy + sd_z_node_offset.at(sdZOffset())) == (z_nbl + m_build_info.m_bloc_nz.at(z_ibl) * all_nb_node_xy))) {
        info() << " Creation hit z bloc boundarz: @ node " << z;
        z_nbl += m_build_info.m_bloc_nz.at(z_ibl) * all_nb_node_xy;
        // On incrémente de bloc
        z_ibl += 1;
        z_obl = 0;
      }
    }
    Real nz = 0.0;
    if (m_mesh_dimension == 3)
      nz = nzDelta(z_obl, z_ibl);
    Integer y_ibl = sd_y_ibl.at(sdYOffset());
    Integer y_obl = sd_y_obl.at(sdYOffset());
    Int64 y_nbl = sd_y_nbl.at(sdYOffset());
    for (Int64 y = 0; y < own_nb_node_y; ++y) {
      // On saute l'éventuelle dernière frontière bl coincidante avec les sd
      // Et si on découvre une frontière bloc, on reset les origines
      if (((y + 1) != own_nb_node_y) &&
          ((y * all_nb_node_x + sd_y_node_offset.at(sdYOffset())) == (y_nbl + m_build_info.m_bloc_ny.at(y_ibl) * all_nb_node_x))) {
        info() << " Creation hit y bloc boundary: @ node " << y;
        y_nbl += m_build_info.m_bloc_ny.at(y_ibl) * all_nb_node_x;
        // On incrémente de bloc
        y_ibl += 1;
        y_obl = 0;
      }
      Real ny = nyDelta(y_obl, y_ibl);
      // Récupération du numéro de bloc courant
      Integer x_ibl = sd_x_ibl.at(sdXOffset());
      // Récupération de l'offset dans le bloc courant
      Integer x_obl = sd_x_obl.at(sdXOffset());
      // Récupération du nombre de noeuds dans le bloc courant
      Int64 x_nbl = sd_x_nbl.at(sdXOffset());
      for (Int64 x = 0; x < own_nb_node_x; ++x) {
        Int64 node_unique_id = node_unique_id_offset + x + y * all_nb_node_x + z * all_nb_node_xy;
        nodes_unique_id[node_local_id] = node_unique_id;
        Int32 owner = m_my_mesh_part;
        // Si on est pas sur sur un sd des bords
        // Et si on touche aux noeuds des bords du sd de bord
        if (((sdXOffset() + 1) != m_build_info.m_nsdx) && ((x + 1) == own_nb_node_x))
          owner += 1;
        if (((sdYOffset() + 1) != m_build_info.m_nsdy) && ((y + 1) == own_nb_node_y))
          owner += m_build_info.m_nsdx;
        if (((sdZOffset() + 1) != m_build_info.m_nsdz) && ((z + 1) == own_nb_node_z))
          owner += m_build_info.m_nsdx * m_build_info.m_nsdy;
        // On saute l'éventuelle dernière frontière bl coincidante avec les sd
        // Et si on découvre une frontière bloc, on reset les origines
        if (((x + 1) != own_nb_node_x) && ((x + sd_x_node_offset.at(sdXOffset())) == (x_nbl + m_build_info.m_bloc_nx.at(x_ibl)))) {
          info() << " Creation hit x bloc boundary: @ node " << x;
          x_nbl += m_build_info.m_bloc_nx.at(x_ibl);
          // On incrémente de bloc
          x_ibl += 1;
          x_obl = 0;
        }
        Real nx = nxDelta(x_obl, x_ibl);
        /*debug() << "[2;33m[CartesianMeshGenerator::generateMesh] node @ "<<x<<"x"<<y<<"x"<<z
               <<":"<< ", uid=" << node_unique_id<< ", owned by " << owner
               << ", [1;32m coords=("<<nx<<","<<ny<<","<<nz<<")"<<"[0m";*/
        nodes_infos.nocheckAdd(node_unique_id, NodeInfo(owner, Real3(nx, ny, nz)));
        node_local_id += 1;
        x_obl += 1;
      }
      y_obl += 1;
    }
    z_obl += 1;
  }

  // Création des mailles
  // Infos pour la création des mailles
  // par maille: 1 pour son unique id,
  //             1 pour son type,
  //             8 pour chaque noeud
  Int64 cell_unique_id_offset = 0;
  // On calcule l'offset en x des cell_unique_id
  cell_unique_id_offset += sd_x_cell_offset.at(sdXOffset());
  info() << "cell_unique_id_offset=" << cell_unique_id_offset;
  // On calcule l'offset en y des cell_unique_id
  cell_unique_id_offset += sd_y_cell_offset.at(sdYOffset());
  info() << "cell_unique_id_offset=" << cell_unique_id_offset;
  if (m_mesh_dimension == 3) {
    // On calcule l'offset en z des cell_unique_id
    cell_unique_id_offset += sd_z_cell_offset.at(sdZOffset());
    info() << "cell_unique_id_offset=" << cell_unique_id_offset;
  }
  info() << "cell_unique_id_offset=" << cell_unique_id_offset;
  m_generation_info->setFirstOwnCellUniqueId(cell_unique_id_offset);

  const Int32 face_numbering_version = m_build_info.m_face_numbering_version;
  info() << "FaceNumberingVersion = " << face_numbering_version;
  const Int32 edge_numbering_version = m_build_info.m_edge_numbering_version;
  info() << "EdgeNumberingVersion = " << edge_numbering_version;

  {
    info() << "Set Specific info for cartesian mesh";
    if (m_mesh_dimension==3)
      cartesian_mesh_build_info.setInfos3D({all_nb_cell_x,all_nb_cell_y,all_nb_cell_z},
                                           {own_nb_cell_x,own_nb_cell_y,own_nb_cell_z},
                                           {first_own_cell_offset.x,first_own_cell_offset.y,first_own_cell_offset.z},
                                           0 );
    else if (m_mesh_dimension==2){
      cartesian_mesh_build_info.setInfos2D({all_nb_cell_x,all_nb_cell_y},
                                           {own_nb_cell_x,own_nb_cell_y},
                                           {first_own_cell_offset.x,first_own_cell_offset.y},
                                           0 );
    }
    else
      ARCANE_FATAL("Invalid dimensionn '{0}' (valid values are 2 or 3)",m_mesh_dimension);

    if (face_numbering_version>=0)
      cartesian_mesh_build_info._internal()->setFaceBuilderVersion(face_numbering_version);
    if (edge_numbering_version>=0)
      cartesian_mesh_build_info._internal()->setEdgeBuilderVersion(edge_numbering_version);
  }

  cartesian_mesh_build_info.allocateMesh();

  VariableNodeReal3& nodes_coord_var(mesh->nodesCoordinates());
  {
    info() << "Fills the variable containing the coordinates of the nodes";
    Int32UniqueArray nodes_local_id(nodes_unique_id.size());
    IItemFamily* family = mesh->nodeFamily();
    family->itemsUniqueIdToLocalId(nodes_local_id, nodes_unique_id);
    NodeInfoListView nodes_internal(family);
    for (Integer i = 0; i < node_local_id; ++i) {
      Node node = nodes_internal[nodes_local_id[i]];
      Int64 unique_id = nodes_unique_id[i];
      nodes_coord_var[node] = nodes_infos.lookupValue(unique_id).m_coord;
      /*debug() << "[2;33m[CartesianMeshGenerator::generateMesh] Set coord "
        << ItemPrinter(node) << " coord=" << nodes_coord_var[node]<< "[0m";*/
    }
  }
  nodes_coord_var.synchronize();

  // Créé les groupes correspondants aux bords du maillage
  // Si demandé, on créé aussi les groupes pour tester un tube à choc de Sod.
  {
    SodStandardGroupsBuilder groups_builder(traceMng());
    Real3 origin = m_build_info.m_origine;
    Real3 length(m_l.x,m_l.y,m_l.z);
    Real3 max_pos = origin + length;
    // TODO: Comme il peut y avoir des progressions geométriques il faut définir
    // le milieu à partir de la position de la maille d'offset le milieu
    // et pas à partir des coordonnées
    // Calculer middle_x comme position du milieu
    bool do_zg_and_zd = m_build_info.m_is_generate_sod_groups;
    Real middle_x = (origin.x + max_pos.x) / 2.0;
    Real middle_height = (origin.y + max_pos.y) / 2.0;
    groups_builder.generateGroups(mesh, origin, origin + length, middle_x, middle_height, do_zg_and_zd);
  }

  return false; // false == ok
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de génération de maillage cartésien en 2D.
 */
class Cartesian2DMeshGenerator
: public ArcaneCartesian2DMeshGeneratorObject
{
 public:

  explicit Cartesian2DMeshGenerator(const ServiceBuildInfo& sbi)
  : ArcaneCartesian2DMeshGeneratorObject(sbi){}

 public:

  void fillMeshBuildInfo(MeshBuildInfo& build_info) override
  {
    ARCANE_UNUSED(build_info);
    info() << "Cartesian2DMeshGenerator: fillMeshBuildInfo()";
    m_build_info.m_nsdx = options()->nbPartX();
    m_build_info.m_nsdy = options()->nbPartY();

    m_build_info.m_mesh_dimension = 2;
    Real2 origin = options()->origin;
    m_build_info.m_origine.x = origin.x;
    m_build_info.m_origine.y = origin.y;
    m_build_info.m_is_generate_sod_groups = options()->generateSodGroups();
    m_build_info.m_face_numbering_version = options()->faceNumberingVersion();

    for( auto& o : options()->x() ){
      m_build_info.m_bloc_lx.add(o->length);
      m_build_info.m_bloc_nx.add(o->n);
      m_build_info.m_bloc_px.add(o->progression);
    }

    for( auto& o : options()->y() ){
      m_build_info.m_bloc_ly.add(o->length);
      m_build_info.m_bloc_ny.add(o->n);
      m_build_info.m_bloc_py.add(o->progression);
    }
  }

  void allocateMeshItems(IPrimaryMesh* pm) override
  {
    info() << "Cartesian2DMeshGenerator: allocateMeshItems()";
    CartesianMeshGenerator g(pm);
    // Regarde s'il faut calculer dynamiquement le découpage
    auto [ x, y ] = _computePartition(pm,m_build_info.m_nsdx,m_build_info.m_nsdy);
    m_build_info.m_nsdx = x;
    m_build_info.m_nsdy = y;
    g.setBuildInfo(m_build_info);
    g.generateMesh();
  }

 private:

  CartesianMeshGeneratorBuildInfo m_build_info;

  static std::tuple<Integer,Integer>
  _computePartition(IPrimaryMesh* pm,Integer nb_x,Integer nb_y)
  {
    Int32 nb_part = pm->meshPartInfo().nbPart();
    // En séquentiel, ne tient pas compte des valeurs de jeu de données.
    if (nb_part==1)
      return {1,1};
    // Si le découpage en X et en Y est spécifié, l'utilise directement.
    if (nb_x!=0 && nb_y!=0)
      return {nb_x, nb_y};
    // Aucun découpage spécifié. Il faut que math::sqrt(nb_part)
    // soit un entier
    if (nb_x==0 && nb_y==0){
      double s = math::sqrt((double)(nb_part));
      Integer s_int = (Integer)(::floor(s));
      if ((s_int*s_int) != nb_part)
        ARCANE_FATAL("Invalid number of part '{0}' for automatic partitioning: sqrt({1}) is not an integer",
                     nb_part,nb_part);
      return {s_int,s_int};
    }
    // Ici, on a un des deux découpages qui n'est pas spécifié.
    if (nb_x==0){
      if ( (nb_part % nb_y) != 0 )
        ARCANE_FATAL("Invalid number of Y part '{0}' for automatic partitioning: can not divide '{1}' by '{2}'",
                     nb_y,nb_part,nb_y);
      nb_x = nb_part / nb_y;
    }
    else{
      if ( (nb_part % nb_x) != 0 )
        ARCANE_FATAL("Invalid number of X part '{0}' for automatic partitioning: can not divide '{1}' by '{2}'",
                     nb_x,nb_part,nb_x);
      nb_y = nb_part / nb_x;
    }
    return {nb_x,nb_y};
  }
};

ARCANE_REGISTER_SERVICE_CARTESIAN2DMESHGENERATOR(Cartesian2D,Cartesian2DMeshGenerator);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de génération de maillage cartésien en en 3D.
 */
class Cartesian3DMeshGenerator
: public ArcaneCartesian3DMeshGeneratorObject
{
 public:
  Cartesian3DMeshGenerator(const ServiceBuildInfo& sbi)
  : ArcaneCartesian3DMeshGeneratorObject(sbi){}
 public:
  void fillMeshBuildInfo(MeshBuildInfo& build_info) override
  {
    ARCANE_UNUSED(build_info);
    info() << "Cartesian3DMeshGenerator: fillMeshBuildInfo()";
    m_build_info.m_nsdx = options()->nbPartX();
    m_build_info.m_nsdy = options()->nbPartY();
    m_build_info.m_nsdz = options()->nbPartZ();

    m_build_info.m_mesh_dimension = 3;
    Real3 origin = options()->origin;
    m_build_info.m_origine.x = origin.x;
    m_build_info.m_origine.y = origin.y;
    m_build_info.m_origine.z = origin.z;
    m_build_info.m_is_generate_sod_groups = options()->generateSodGroups();
    m_build_info.m_face_numbering_version = options()->faceNumberingVersion();
    m_build_info.m_edge_numbering_version = options()->edgeNumberingVersion();

    for( auto& o : options()->x() ){
      m_build_info.m_bloc_lx.add(o->length);
      m_build_info.m_bloc_nx.add(o->n);
      m_build_info.m_bloc_px.add(o->progression);
    }

    for( auto& o : options()->y() ){
      m_build_info.m_bloc_ly.add(o->length);
      m_build_info.m_bloc_ny.add(o->n);
      m_build_info.m_bloc_py.add(o->progression);
    }

    for( auto& o : options()->z() ){
      m_build_info.m_bloc_lz.add(o->length);
      m_build_info.m_bloc_nz.add(o->n);
      m_build_info.m_bloc_pz.add(o->progression);
    }
  }

  void allocateMeshItems(IPrimaryMesh* pm) override
  {
    info() << "Cartesian3DMeshGenerator: allocateMeshItems()";
    CartesianMeshGenerator g(pm);
    g.setBuildInfo(m_build_info);
    g.generateMesh();
  }

  CartesianMeshGeneratorBuildInfo m_build_info;
};

ARCANE_REGISTER_SERVICE_CARTESIAN3DMESHGENERATOR(Cartesian3D,Cartesian3DMeshGenerator);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
