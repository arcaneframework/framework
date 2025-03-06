// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MshMeshWriter.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Lecture/Écriture d'un fichier au format MSH.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IOException.h"
#include "arcane/utils/FixedArray.h"

#include "arcane/core/FactoryService.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/AbstractService.h"
#include "arcane/core/IMeshWriter.h"
#include "arcane/core/ItemTypeMng.h"
#include "arcane/core/SharedVariable.h"

#include "arcane/std/internal/IosGmsh.h"

#include <tuple>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Écriture des fichiers de maillage au format msh.
 */
class MshMeshWriter
: public AbstractService
, public IMeshWriter
{
 public:

  struct EntityInfo
  {
    Int32 m_dim = -1;
    Int32 m_item_type = IT_NullType;
    Int32 m_entity_tag = -1;
  };

 public:

  explicit MshMeshWriter(const ServiceBuildInfo& sbi);

 public:

  void build() override {}
  bool writeMeshToFile(IMesh* mesh, const String& file_name) override;

 private:

  ItemTypeMng* m_item_type_mng = nullptr;

 private:

  bool _writeMeshToFileV4(IMesh* mesh, const String& file_name);
  Integer _convertToMshType(Int32 arcane_type);
  std::pair<Int64, Int64> _getFamilyMinMaxUniqueId(IItemFamily* family);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(MshMeshWriter, IMeshWriter, MshNewMeshWriter);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MshMeshWriter::
MshMeshWriter(const ServiceBuildInfo& sbi)
: AbstractService(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti le type %Arcane en le type GMSH.
 *
 * La conversion n'est possible que pour les types de base.
 */
Integer MshMeshWriter::
_convertToMshType(Int32 arcane_type)
{
  switch (arcane_type) {
    //		case (IT_NullType):			return MSH_LIN_2;		//case (0) is not used
  case IT_Vertex:
    return MSH_PNT; //printf("1-node point");
  case IT_Cell3D_Line2:
  case IT_Line2:
    return MSH_LIN_2; //printf("2-node line");
  case IT_Cell3D_Triangle3:
  case IT_Triangle3:
    return MSH_TRI_3; //printf("3-node triangle");
  case IT_Cell3D_Quad4:
  case IT_Quad4:
    return MSH_QUA_4; //printf("4-node quadrangle");
  case IT_Tetraedron4:
    return MSH_TET_4; //printf("4-node tetrahedron");
  case IT_Hexaedron8:
    return MSH_HEX_8; //printf("8-node hexahedron");
  case IT_Pentaedron6:
    return MSH_PRI_6; //printf("6-node prism");
  case IT_Pyramid5:
    return MSH_PYR_5; //printf("5-node pyramid");
  // Beneath, are some meshes that have been tried to match gmsh's ones
  // Other 5-nodes
  case IT_Pentagon5:
    return MSH_PYR_5; // Could use a tag to encode these
  case IT_HemiHexa5:
    return MSH_PYR_5;
  case IT_DiTetra5:
    return MSH_PYR_5;
  // Other 6-nodes
  case IT_Hexagon6:
    return MSH_PRI_6;
  case IT_HemiHexa6:
    return MSH_PRI_6;
  case IT_AntiWedgeLeft6:
    return MSH_PRI_6;
  case IT_AntiWedgeRight6:
    return MSH_PRI_6;
  // Other 10-nodes
  case IT_Heptaedron10:
    return MSH_TRI_10;
  // Other 12-nodes
  case IT_Octaedron12:
    return MSH_TRI_12;
  // Others ar still considered as default, rising an exception
  default:
    break;
  }
  ARCANE_THROW(NotSupportedException, "Arcane type '{0}'", m_item_type_mng->typeFromId(arcane_type));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MshMeshWriter::
writeMeshToFile(IMesh* mesh, const String& file_name)
{
  return _writeMeshToFileV4(mesh, file_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecrit au format MSH V4.
 *
 * \param mesh Maillage d'entrée
 * \param file_name Nom du fichier de sortie
 * \retval true pour toute erreur détectée
 * \retval false sinon
 */
bool MshMeshWriter::
_writeMeshToFileV4(IMesh* mesh, const String& file_name)
{
  m_item_type_mng = mesh->itemTypeMng();
  String mshFileName(file_name + ".msh");
  std::ofstream ofile(mshFileName.localstr());
  ofile.precision(20);
  if (!ofile)
    throw IOException("VtkMeshIOService::writeMeshToFile(): Unable to open file");

  info() << "[writNodes=" << mesh->nbNode() << " nCells=" << mesh->nbCell();

  ofile << "$MeshFormat\n";
  // 4.1 pour le format
  // 0 pour ASCII (1 pour binaire)
  // 8 pour sizeof(size_t)
  ofile << "4.1 0 " << sizeof(size_t) << "\n";
  ofile << "$EndMeshFormat\n";

  IItemFamily* cell_family = mesh->cellFamily();
  CellGroup all_cells = mesh->allCells();
  const Int32 mesh_nb_node = mesh->nbNode();
  IItemFamily* node_family = mesh->nodeFamily();
  ItemGroup all_nodes = mesh->allNodes();

  // Pour GMSH, il faut commencer par les 'Entities'.
  // Il faut une entité par type de maille.
  // On commence donc par calculer les types de mailles.
  // Pour les maillages non-manifold les mailles peuvent être de dimension
  // différentes
  FixedArray<EntityInfo, NB_BASIC_ITEM_TYPE> entities_by_type;

  // Pour GMSH, il faut trier les mailles par leur type (Triangle, Quadrangle, ...)
  // On fait une entity par type de maille.
  FixedArray<UniqueArray<Int32>, NB_BASIC_ITEM_TYPE> cells_by_type;
  ENUMERATE_ (Cell, icell, all_cells) {
    Cell cell = *icell;
    Int16 cell_type = cell.type();
    if (cell_type >= NB_BASIC_ITEM_TYPE || cell_type <= 0)
      ARCANE_FATAL("Only pre-defined cell type are supported (current cell type is '{0}')",
                   m_item_type_mng->typeFromId(cell_type));
    cells_by_type[cell_type].add(cell.localId());
  }

  FixedArray<Int32, 4> nb_entities_by_dim;

  // Conserve les types pré-définis qui ont des éléments
  UniqueArray<Int32> existing_cells_type;
  Int64 total_nb_cell = 0;
  for (Int32 i = 0; i < NB_BASIC_ITEM_TYPE; ++i) {
    Int64 nb_type = cells_by_type[i].size();
    if (nb_type > 0)
      existing_cells_type.add(i);
    total_nb_cell += nb_type;
  }
  Int32 nb_existing_type = existing_cells_type.size();
  info() << "NbExistingType=" << nb_existing_type;
  for (Int32 type_index = 0; type_index < nb_existing_type; ++type_index) {
    Int32 cell_type = existing_cells_type[type_index];
    ItemTypeInfo* item_type_info = m_item_type_mng->typeFromId(cell_type);
    Int32 type_dimension = item_type_info->dimension();
    entities_by_type[cell_type] = EntityInfo{ type_dimension, cell_type, (type_index + 1) * 100 };
    ++nb_entities_by_dim[type_dimension];
  }

  // Calcule la bounding box des noeuds
  VariableNodeReal3 nodes_coords = mesh->nodesCoordinates();
  Real3 node_min_bounding_box;
  Real3 node_max_bounding_box;
  {
    Real max_value = FloatInfo<Real>::maxValue();
    Real min_value = -max_value;
    Real3 min_box(max_value, max_value, max_value);
    Real3 max_box(min_value, min_value, min_value);
    ENUMERATE_ (Node, inode, all_nodes) {
      Real3 pos = nodes_coords[inode];
      min_box = math::min(min_box, pos);
      max_box = math::max(max_box, pos);
    }
    node_min_bounding_box = min_box;
    node_max_bounding_box = max_box;
  }

  // $Entities
  // numPoints(size_t) numCurves(size_t)
  //   numSurfaces(size_t) numVolumes(size_t)
  // pointTag(int) X(double) Y(double) Z(double)
  //   numPhysicalTags(size_t) physicalTag(int) ...
  // ...
  // curveTag(int) minX(double) minY(double) minZ(double)
  //   maxX(double) maxY(double) maxZ(double)
  //   numPhysicalTags(size_t) physicalTag(int) ...
  //   numBoundingPoints(size_t) pointTag(int; sign encodes orientation) ...
  // ...
  // surfaceTag(int) minX(double) minY(double) minZ(double)
  //   maxX(double) maxY(double) maxZ(double)
  //   numPhysicalTags(size_t) physicalTag(int) ...
  //   numBoundingCurves(size_t) curveTag(int; sign encodes orientation) ...
  // ...
  // volumeTag(int) minX(double) minY(double) minZ(double)
  //   maxX(double) maxY(double) maxZ(double)
  //   numPhysicalTags(size_t) physicalTag(int) ...
  //   numBoundngSurfaces(size_t) surfaceTag(int; sign encodes orientation) ...
  // ...
  // $EndEntities
  {
    ofile << "$Entities\n";
    nb_entities_by_dim[0] = 0;
    ofile << nb_entities_by_dim[0] << " " << nb_entities_by_dim[1]
          << " " << nb_entities_by_dim[2] << " " << nb_entities_by_dim[3] << "\n";
    for (Int32 idim = 1; idim < 4; ++idim) {
      for (Int32 k = 0; k < NB_BASIC_ITEM_TYPE; ++k) {
        const EntityInfo& entity_info = entities_by_type[k];
        if (entity_info.m_dim != idim)
          continue;
        ofile << entity_info.m_entity_tag << " " << node_min_bounding_box.x << " " << node_min_bounding_box.y << " " << node_min_bounding_box.z
              << " " << node_max_bounding_box.x << " " << node_max_bounding_box.y << " " << node_max_bounding_box.z;
        // Pas de tag pour l'instant
        ofile << " 0";
        // Pas de bounding pour l'instant
        ofile << " 0";
        ofile << "\n";
      }
    }
    ofile << "$EndEntities\n";
  }

  // $Nodes
  // numEntityBlocks(size_t) numNodes(size_t)
  //   minNodeTag(size_t) maxNodeTag(size_t)
  // entityDim(int) entityTag(int) parametric(int; 0 or 1)
  //   numNodesInBlock(size_t)
  //   nodeTag(size_t)
  //   ...
  //   x(double) y(double) z(double)
  //      < u(double; if parametric and entityDim >= 1) >
  //      < v(double; if parametric and entityDim >= 2) >
  //      < w(double; if parametric and entityDim == 3) >
  //   ...
  // ...
  // $EndNodes

  // Bloc contenant les noeuds
  ofile << "$Nodes\n";

  auto [node_min_uid, node_max_uid] = _getFamilyMinMaxUniqueId(node_family);

  ofile << "1 " << mesh_nb_node << " " << node_min_uid << " " << node_max_uid << "\n";
  // entityDim(int) entityTag(int) parametric(int; 0 or 1) numNodesInBlock(size_t)
  ofile << "0 " << "100 " << "0 " << mesh_nb_node << "\n";

  // Sauve les uniqueId()
  ENUMERATE_ (Node, inode, all_nodes) {
    Int64 uid = inode->uniqueId();
    ofile << uid << "\n";
  }

  // Sauve les coordonnées
  ENUMERATE_ (Node, inode, all_nodes) {
    Real3 coord = nodes_coords[inode];
    ofile << coord.x << " " << coord.y << " " << coord.z << "\n";
  }

  ofile << "$EndNodes\n";

  auto [cell_min_uid, cell_max_uid] = _getFamilyMinMaxUniqueId(cell_family);

  // $Elements
  //   numEntityBlocks(size_t) numElements(size_t)
  //     minElementTag(size_t) maxElementTag(size_t)
  //   entityDim(int) entityTag(int) elementType(int; see below)
  //     numElementsInBlock(size_t)
  //     elementTag(size_t) nodeTag(size_t) ...
  //     ...
  //   ...
  // $EndElements

  // Bloc contenant les mailles
  ofile << "$Elements\n";

  ofile << nb_existing_type << " " << total_nb_cell << " " << cell_min_uid << " " << cell_max_uid << "\n";
  for (Int32 type_index = 0; type_index < nb_existing_type; ++type_index) {
    Int32 cell_type = existing_cells_type[type_index];
    const EntityInfo& entity_info = entities_by_type[cell_type];
    ConstArrayView<Int32> cells_of_current_type = cells_by_type[cell_type];
    ItemTypeInfo* item_type_info = m_item_type_mng->typeFromId(cell_type);
    Int32 type_dimension = item_type_info->dimension();
    ofile << "\n";
    ofile << type_dimension << " " << entity_info.m_entity_tag << " " << _convertToMshType(cell_type)
          << " " << cells_of_current_type.size() << "\n";
    info() << "Writing cells of type=" << item_type_info->typeName()
           << " n=" << cells_of_current_type.size()
           << " dimension=" << type_dimension;
    Int32 nb_node_for_type = item_type_info->nbLocalNode();
    ENUMERATE_ (Cell, icell, cell_family->view(cells_of_current_type)) {
      Cell cell = *icell;
      ofile << cell.uniqueId();
      for (Int32 i = 0; i < nb_node_for_type; ++i)
        ofile << " " << cell.node(i).uniqueId();
      ofile << "\n";
    }
  }
  ofile << "$EndElements\n";

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<Int64, Int64> MshMeshWriter::
_getFamilyMinMaxUniqueId(IItemFamily* family)
{
  Int64 min_uid = INT64_MAX;
  Int64 max_uid = -1;
  ENUMERATE_ (Item, iitem, family->allItems()) {
    Item item = *iitem;
    Int64 uid = item.uniqueId();
    if (uid < min_uid)
      min_uid = uid;
    if (uid > max_uid)
      max_uid = uid;
  }
  return { min_uid, max_uid };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
