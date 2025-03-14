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
#include "arcane/utils/Collection.h"
#include "arcane/utils/ITraceMng.h"

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

  class ItemFamilyWriteInfo
  : public TraceAccessor
  {
   public:

    explicit ItemFamilyWriteInfo(ITraceMng* tm)
    : TraceAccessor(tm)
    {
    }
  };

  struct PhysicalTagInfo
  {
   public:

    Int32 m_dimension = -1;
    Int32 m_physical_tag = -1;
    String m_name;
  };
  struct EntityInfo
  {
   public:

    EntityInfo(Int32 dim, Int32 item_type, Int32 entity_tag)
    : m_dim(dim)
    , m_item_type(item_type)
    , m_entity_tag(entity_tag)
    {
    }

   public:

    void setPhysicalTag(Int32 tag, const String& name)
    {
      m_physical_tag = tag;
      m_physical_tag_name = name;
    }

   public:

    Int32 m_dim = -1;
    Int32 m_item_type = IT_NullType;
    Int32 m_entity_tag = -1;
    Int32 m_physical_tag = -1;
    String m_physical_tag_name;
  };

  class ItemGroupWriteInfo
  {
   public:

    void processGroup(ItemGroup group, Int32 base_entity_index);

   public:

    ConstArrayView<EntityInfo> entitiesByType() const { return m_entities_by_type; }
    ConstArrayView<Int32> itemsByType(Int32 item_type) const { return m_items_by_type[item_type]; }

   private:

    UniqueArray<EntityInfo> m_entities_by_type;
    FixedArray<UniqueArray<Int32>, NB_BASIC_ITEM_TYPE> m_items_by_type;
    UniqueArray<Int32> m_existing_items_type;
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
  ARCANE_THROW(NotSupportedException, "Arcane type '{0}'", m_item_type_mng->typeFromId(arcane_type)->typeName());
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

void MshMeshWriter::ItemGroupWriteInfo::
processGroup(ItemGroup group, Int32 base_entity_index)
{
  String group_name = group.name();
  bool is_all_items = group.isAllItems();
  IItemFamily* family = group.itemFamily();
  ItemTypeMng* item_type_mng = family->mesh()->itemTypeMng();
  ITraceMng* tm = family->traceMng();

  // Pour GMSH, il faut trier les mailles par leur type (Triangle, Quadrangle, ...)
  // On fait une entity MSH par type d'entité Arcane.

  ENUMERATE_ (Item, iitem, group) {
    Item item = *iitem;
    Int16 item_type = item.type();
    if (item_type >= NB_BASIC_ITEM_TYPE || item_type <= 0)
      ARCANE_FATAL("Only pre-defined Item type are supported (current item type is '{0}')",
                   item_type_mng->typeFromId(item_type)->typeName());
    m_items_by_type[item_type].add(item.localId());
  }

  // Conserve les types pré-définis qui ont des éléments
  Int64 total_nb_item = 0;
  for (Int32 i = 0; i < NB_BASIC_ITEM_TYPE; ++i) {
    Int64 nb_type = m_items_by_type[i].size();
    if (nb_type > 0)
      m_existing_items_type.add(i);
    total_nb_item += nb_type;
  }

  Int32 nb_existing_type = m_existing_items_type.size();
  tm->info() << "NbExistingType=" << nb_existing_type;
  for (Int32 type_index = 0; type_index < nb_existing_type; ++type_index) {
    Int32 item_type = m_existing_items_type[type_index];
    ItemTypeInfo* item_type_info = item_type_mng->typeFromId(item_type);
    Int32 type_dimension = item_type_info->dimension();
    EntityInfo entity_info(type_dimension, item_type, base_entity_index + type_index);
    if (!is_all_items)
      entity_info.setPhysicalTag(base_entity_index + type_index, group_name);
    m_entities_by_type.add(entity_info);
    //++nb_entities_by_dim[type_dimension];
  }
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
  String mesh_file_name(file_name);
  if (!file_name.endsWith(".msh"))
    mesh_file_name = mesh_file_name + ".msh";
  std::ofstream ofile(mesh_file_name.localstr());
  ofile.precision(20);
  if (!ofile)
    ARCANE_THROW(IOException, "Unable to open file '{0}' for writing", mesh_file_name);

  info() << "writing file '" << mesh_file_name << "'";

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

  UniqueArray<ItemGroup> items_groups;
  // Parcours tous les groupes
  // NOTE: Pour l'instant on suppose que cela forme une partition
  // du maillage.
  for (ItemGroup group : cell_family->groups()) {
    if (group.isAllItems())
      continue;
    if (group.isAutoComputed())
      continue;
    info() << "Processing ItemGroup group=" << group.name() << " family=" << group.itemFamily()->name();
    items_groups.add(group);
  }
  // Si pas de groupes, on prend celui de toutes les mailles
  // TODO: toujours prendre ce groupe pour les entités qui n'auraient pas
  // été traitées par les autres groupes lorsqu'on n'est pas sur
  // que l'ensemble des groupes forme une partition.
  if (items_groups.empty())
    items_groups.add(cell_family->allItems());

  std::vector<std::unique_ptr<ItemGroupWriteInfo>> groups_write_info_list;
  Int32 base_entity_index = 1000;
  for (ItemGroup group : items_groups) {
    auto x(std::make_unique<ItemGroupWriteInfo>());
    x->processGroup(group, base_entity_index);
    groups_write_info_list.emplace_back(std::move(x));
    base_entity_index += 1000;
  }

  // Pour GMSH, il faut commencer par les 'Entities'.
  // Il faut une entité par type de maille.
  // On commence donc par calculer les types de mailles.
  // Pour les maillages non-manifold les mailles peuvent être de dimension
  // différentes

  // Liste des tags physiques
  UniqueArray<PhysicalTagInfo> physical_tags;

  // Calcule le nombre d'entités par dimension
  FixedArray<Int32, 4> nb_entities_by_dim;

  // Calcule le nombre total d'éléments.
  // Toutes les entités qui ne sont pas de dimension 0 sont des éléments.
  Int64 total_nb_cell = 0;
  for (const auto& ginfo : groups_write_info_list) {
    for (const EntityInfo& entity_info : ginfo->entitiesByType()) {
      Int32 dim = entity_info.m_dim;
      if (dim >= 0)
        ++nb_entities_by_dim[dim];
      if (dim > 0) {
        Int32 item_type = entity_info.m_item_type;
        Int32 nb_item = ginfo->itemsByType(item_type).size();
        total_nb_cell += nb_item;

        Int32 physical_tag = entity_info.m_physical_tag;
        if (physical_tag > 0) {
          physical_tags.add(PhysicalTagInfo{ dim, physical_tag, entity_info.m_physical_tag_name });
        }
      }
    }
  }

  // $PhysicalNames // same as MSH version 2
  //   numPhysicalNames(ASCII int)
  //   dimension(ASCII int) physicalTag(ASCII int) "name"(127 characters max)
  //   ...
  // $EndPhysicalNames

  {
    ofile << "$PhysicalNames\n";
    Int32 nb_tag = physical_tags.size();
    ofile << nb_tag << "\n";
    for (const PhysicalTagInfo& tag_info : physical_tags) {
      // TODO: vérifier que le nom ne dépasse pas 127 caractères.
      ofile << tag_info.m_dimension << " " << tag_info.m_physical_tag << " " << tag_info.m_name << "\n";
    }
    ofile << "$EndPhysicalNames\n";
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
    //nb_entities_by_dim[0] = 0;
    ofile << nb_entities_by_dim[0] << " " << nb_entities_by_dim[1]
          << " " << nb_entities_by_dim[2] << " " << nb_entities_by_dim[3] << "\n";
    for (Int32 idim = 1; idim < 4; ++idim) {
      for (const auto& ginfo : groups_write_info_list) {
        for (const EntityInfo& entity_info : ginfo->entitiesByType()) {
          if (entity_info.m_dim != idim)
            continue;
          ofile << entity_info.m_entity_tag << " " << node_min_bounding_box.x << " " << node_min_bounding_box.y << " " << node_min_bounding_box.z
                << " " << node_max_bounding_box.x << " " << node_max_bounding_box.y << " " << node_max_bounding_box.z;
          // Pas de tag pour l'instant
          Int32 physical_tag = entity_info.m_physical_tag;
          if (physical_tag > 0) {
            ofile << " 1 " << physical_tag;
          }
          else
            ofile << " 0";
          // Pas de boundary pour l'instant
          ofile << " 0";
          ofile << "\n";
        }
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

  Int32 nb_existing_type = nb_entities_by_dim[1] + nb_entities_by_dim[2] + nb_entities_by_dim[3];
  ofile << nb_existing_type << " " << total_nb_cell << " " << cell_min_uid << " " << cell_max_uid << "\n";
  for (const auto& ginfo : groups_write_info_list) {
    for (const EntityInfo& entity_info : ginfo->entitiesByType()) {
      //for (Int32 type_index = 0; type_index < nb_existing_type; ++type_index) {
      Int32 cell_type = entity_info.m_item_type; //existing_cells_type[type_index];
      //const EntityInfo& entity_info = entities_by_type[cell_type];
      ConstArrayView<Int32> cells_of_current_type = ginfo->itemsByType(cell_type);
      ItemTypeInfo* item_type_info = m_item_type_mng->typeFromId(cell_type);
      Int32 type_dimension = entity_info.m_dim;
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
