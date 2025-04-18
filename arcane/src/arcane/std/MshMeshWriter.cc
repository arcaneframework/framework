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
#include "arcane/core/internal/MshMeshGenerationInfo.h"

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
: public TraceAccessor
{
  using MshPeriodicOneInfo = impl::MshMeshGenerationInfo::MshPeriodicOneInfo;

 public:

  //! Informations de correspondance entre le type MSH et le type Arcane
  class ArcaneToMshTypeInfo
  {
   public:

    ArcaneToMshTypeInfo() = default;
    ArcaneToMshTypeInfo(ItemTypeId iti, Int32 msh_type, ConstArrayView<Int16> reorder_infos)
    : m_arcane_type(iti)
    , m_msh_type(msh_type)
    , m_reorder_infos(reorder_infos)
    {
    }

   public:

    ItemTypeId m_arcane_type;
    Int32 m_msh_type = -1;
    UniqueArray<Int16> m_reorder_infos;
  };

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

    EntityInfo(Int32 dim, ItemTypeId item_type, Int32 entity_tag)
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
    ItemTypeId m_item_type;
    Int32 m_entity_tag = -1;
    Int32 m_physical_tag = -1;
    String m_physical_tag_name;
  };

  class ItemGroupWriteInfo
  {
   public:

    void processGroup(ItemGroup group, Int32 base_entity_index);

   public:

    const ItemGroup& group() const { return m_item_group; }
    ConstArrayView<EntityInfo> entitiesByType() const { return m_entities_by_type; }
    ConstArrayView<Int32> itemsByType(Int32 item_type) const { return m_items_by_type[item_type]; }

   private:

    ItemGroup m_item_group;
    UniqueArray<EntityInfo> m_entities_by_type;
    FixedArray<UniqueArray<Int32>, NB_BASIC_ITEM_TYPE> m_items_by_type;
    UniqueArray<ItemTypeId> m_existing_items_type;
  };

 public:

  explicit MshMeshWriter(IMesh* mesh);

 public:

  void writeMesh(const String& file_name);

 private:

  IMesh* m_mesh = nullptr;
  ItemTypeMng* m_item_type_mng = nullptr;

  // Liste des tags physiques
  UniqueArray<PhysicalTagInfo> m_physical_tags;

  // Nombre d'entités par dimension
  FixedArray<Int32, 4> m_nb_entities_by_dim;

  //! Liste des informations à écrire pour chaque groupe
  std::vector<std::unique_ptr<ItemGroupWriteInfo>> m_groups_write_info_list;

  impl::MshMeshGenerationInfo* m_mesh_info = nullptr;
  bool m_has_periodic_info = false;

  //! Informations de conversion entre les types Arcane et MSH pour les entités
  UniqueArray<ArcaneToMshTypeInfo> m_arcane_to_msh_type_infos;

 private:

  bool _writeMeshToFileV4(IMesh* mesh, const String& file_name);
  std::pair<Int64, Int64> _getFamilyMinMaxUniqueId(IItemFamily* family);
  void _addGroupsToProcess(IItemFamily* family, Array<ItemGroup>& items_groups);
  void _writeEntities(std::ostream& ofile);
  void _writeNodes(std::ostream& ofile);
  void _writeElements(std::ostream& ofile, Int64 total_nb_cell);
  void _writePeriodic(std::ostream& ofile);
  void _initTypes();
  void _addArcaneTypeInfo(ItemTypeId arcane_type, Int32 msh_type, ConstArrayView<Int16> reorder_infos = {});
  const ArcaneToMshTypeInfo& arcaneToMshTypeInfo(ItemTypeId arcane_type) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MshMeshWriter::
MshMeshWriter(IMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
{
  _initTypes();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshMeshWriter::ItemGroupWriteInfo::
processGroup(ItemGroup group, Int32 base_entity_index)
{
  m_item_group = group;
  String group_name = group.name();
  bool is_all_items = group.isAllItems();
  IItemFamily* family = group.itemFamily();
  IMesh* mesh = family->mesh();
  ItemTypeMng* item_type_mng = mesh->itemTypeMng();
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
  for (Int16 i = 0; i < NB_BASIC_ITEM_TYPE; ++i) {
    Int64 nb_type = m_items_by_type[i].size();
    if (nb_type > 0)
      m_existing_items_type.add(ItemTypeId(i));
    total_nb_item += nb_type;
  }

  Int32 nb_existing_type = m_existing_items_type.size();
  tm->info() << "NbExistingType=" << nb_existing_type;
  for (Int32 type_index = 0; type_index < nb_existing_type; ++type_index) {
    ItemTypeId item_type = m_existing_items_type[type_index];
    ItemTypeInfo* item_type_info = item_type_mng->typeFromId(item_type);
    Int32 type_dimension = item_type_info->dimension();
    EntityInfo entity_info(type_dimension, item_type, base_entity_index + type_index);
    if (!is_all_items)
      entity_info.setPhysicalTag(base_entity_index + type_index, group_name);
    m_entities_by_type.add(entity_info);
  }

  // Si le groupe est vide, il faut quand même un tag physique pour que le
  // groupe soit créé en lecture et ainsi en parallèle garantir que tous
  // les sous-domaines ont les mêmes groupes.
  if (nb_existing_type == 0 && !is_all_items) {
    Int32 mesh_dim = mesh->dimension();
    eItemKind ik = family->itemKind();
    Int32 entity_dim = -1;
    if (ik == IK_Cell)
      entity_dim = mesh_dim;
    else if (ik == IK_Face)
      entity_dim = mesh_dim - 1;
    else if (ik == IK_Edge)
      entity_dim = mesh_dim - 2;
    else
      ARCANE_FATAL("Invalid item kind '{0}' for entity dimension", entity_dim);
    // TODO: prendre un type qui correspond à la dimension
    EntityInfo entity_info(entity_dim, ITI_Tetraedron4, base_entity_index);
    entity_info.setPhysicalTag(base_entity_index, group_name);
    m_entities_by_type.add(entity_info);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Détermine la liste des groupes à traiter pour une famille.
 *
 * La liste des groupes à sauver sera ajoutée à \a items_groups.
 * \note Pour l'instant on support que l'ensemble des groupes forme
 * une partition des entités de la famille.
 */
void MshMeshWriter::
_addGroupsToProcess(IItemFamily* family, Array<ItemGroup>& items_groups)
{
  bool has_group = false;
  // Parcours tous les groupes de la famille
  for (ItemGroup group : family->groups()) {
    if (group.isAllItems())
      continue;
    if (group.isAutoComputed())
      continue;
    info() << "Processing ItemGroup group=" << group.name() << " family=" << group.itemFamily()->name();
    items_groups.add(group);
    has_group = true;
  }
  // Si pas de groupes dans la famille, on prend celui de toutes les entités
  // si la famille est celle des mailles.
  if (!has_group && (family->itemKind() == IK_Cell))
    items_groups.add(family->allItems());

  // TODO: si les groupe traités ne forment pas une partition et qu'il y a
  // des entités qui ne sont pas dans ces groupes, il faudrait tout de même
  // les sauver sous la forme d'une $Entity sans groupe physique associé.
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
void MshMeshWriter::
writeMesh(const String& file_name)
{
  IMesh* mesh = m_mesh;
  m_item_type_mng = mesh->itemTypeMng();
  String mesh_file_name(file_name);
  if (!file_name.endsWith(".msh"))
    mesh_file_name = mesh_file_name + ".msh";
  std::ofstream ofile(mesh_file_name.localstr());
  ofile.precision(20);
  if (!ofile)
    ARCANE_THROW(IOException, "Unable to open file '{0}' for writing", mesh_file_name);

  info() << "writing file '" << mesh_file_name << "'";

  m_mesh_info = impl::MshMeshGenerationInfo::getReference(mesh, false);
  if (m_mesh_info) {
    m_has_periodic_info = m_mesh_info->m_periodic_info.hasValues();
    info() << "Mesh has 'MSH' generation info has_periodic=" << m_has_periodic_info;
  }

  ofile << "$MeshFormat\n";
  // 4.1 pour le format
  // 0 pour ASCII (1 pour binaire)
  // 8 pour sizeof(size_t)
  ofile << "4.1 0 " << sizeof(size_t) << "\n";
  ofile << "$EndMeshFormat\n";

  IItemFamily* cell_family = mesh->cellFamily();
  IItemFamily* face_family = mesh->faceFamily();
  CellGroup all_cells = mesh->allCells();

  UniqueArray<ItemGroup> items_groups;
  _addGroupsToProcess(cell_family, items_groups);
  _addGroupsToProcess(face_family, items_groups);

  const Int32 entity_index_increment = 1000;
  Int32 base_entity_index = entity_index_increment;
  for (ItemGroup group : items_groups) {
    auto x(std::make_unique<ItemGroupWriteInfo>());
    x->processGroup(group, base_entity_index);
    m_groups_write_info_list.emplace_back(std::move(x));
    base_entity_index += entity_index_increment;
  }

  // Pour GMSH, il faut commencer par les 'Entities'.
  // Il faut une entité par type de maille.
  // On commence donc par calculer les types de mailles.
  // Pour les maillages non-manifold les mailles peuvent être de dimension
  // différentes

  // Calcule le nombre total d'éléments.
  // Toutes les entités qui ne sont pas de dimension 0 sont des éléments.
  Int64 total_nb_cell = 0;
  for (const auto& ginfo : m_groups_write_info_list) {
    for (const EntityInfo& entity_info : ginfo->entitiesByType()) {
      Int32 dim = entity_info.m_dim;
      if (dim >= 0)
        ++m_nb_entities_by_dim[dim];
      if (dim > 0) {
        Int32 item_type = entity_info.m_item_type;
        Int32 nb_item = ginfo->itemsByType(item_type).size();
        total_nb_cell += nb_item;

        Int32 physical_tag = entity_info.m_physical_tag;
        if (physical_tag > 0) {
          m_physical_tags.add(PhysicalTagInfo{ dim, physical_tag, entity_info.m_physical_tag_name });
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
    Int32 nb_tag = m_physical_tags.size();
    ofile << nb_tag << "\n";
    for (const PhysicalTagInfo& tag_info : m_physical_tags) {
      // TODO: vérifier que le nom ne dépasse pas 127 caractères.
      ofile << tag_info.m_dimension << " " << tag_info.m_physical_tag << " " << '"' << tag_info.m_name << '"' << "\n";
    }
    ofile << "$EndPhysicalNames\n";
  }

  _writeEntities(ofile);
  _writeNodes(ofile);
  _writeElements(ofile, total_nb_cell);
  _writePeriodic(ofile);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Écrit le bloc contenant les entités ($Entities).
 */
void MshMeshWriter::
_writeEntities(std::ostream& ofile)
{
  ItemGroup all_nodes = m_mesh->allNodes();

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

  // On a besoin de la bouding box de chaque entity.
  // Pour faire simple, on prend celle de tout le maillage mais à terme
  // ce serait mieux de calculer directement la bonne valeur.
  const VariableNodeReal3& nodes_coords = m_mesh->nodesCoordinates();
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
  if (m_has_periodic_info)
    m_nb_entities_by_dim[0] = 1;
  {
    ofile << "$Entities\n";
    ofile << m_nb_entities_by_dim[0] << " " << m_nb_entities_by_dim[1]
          << " " << m_nb_entities_by_dim[2] << " " << m_nb_entities_by_dim[3] << "\n";

    // Si on a des informations de périodicité,
    // on créé une entité de dimension 0 pour que les informations de
    // périodicité y fassent référence. On lui donne le tag 1.
    if (m_has_periodic_info) {
      ofile << "1 0.0 0.0 0.0 0\n";
    }

    for (Int32 idim = 1; idim < 4; ++idim) {
      for (const auto& ginfo : m_groups_write_info_list) {
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
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Écrit le bloc contenant les noeuds ($Nodes).
 */
void MshMeshWriter::
_writeNodes(std::ostream& ofile)
{
  const Int32 mesh_nb_node = m_mesh->nbNode();
  IItemFamily* node_family = m_mesh->nodeFamily();
  ItemGroup all_nodes = m_mesh->allNodes();

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

  // Sauve les uniqueId() des noeuds
  ENUMERATE_ (Node, inode, all_nodes) {
    Int64 uid = inode->uniqueId();
    ofile << uid << "\n";
  }

  // Sauve les coordonnées
  VariableNodeReal3& nodes_coords = m_mesh->nodesCoordinates();
  ENUMERATE_ (Node, inode, all_nodes) {
    Real3 coord = nodes_coords[inode];
    ofile << coord.x << " " << coord.y << " " << coord.z << "\n";
  }

  ofile << "$EndNodes\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Écrit le bloc contenant les élements ($Elements).
 */
void MshMeshWriter::
_writeElements(std::ostream& ofile, Int64 total_nb_cell)
{
  IItemFamily* cell_family = m_mesh->cellFamily();

  // TODO: regarder s'il faut prendre en compte les uniqueId() des faces.
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

  Int32 nb_existing_type = m_nb_entities_by_dim[1] + m_nb_entities_by_dim[2] + m_nb_entities_by_dim[3];
  ofile << nb_existing_type << " " << total_nb_cell << " " << cell_min_uid << " " << cell_max_uid << "\n";
  for (const auto& ginfo : m_groups_write_info_list) {
    ItemGroup item_group = ginfo->group();
    IItemFamily* item_family = item_group.itemFamily();
    for (const EntityInfo& entity_info : ginfo->entitiesByType()) {
      ItemTypeId cell_type = entity_info.m_item_type;
      ConstArrayView<Int32> items_of_current_type = ginfo->itemsByType(cell_type);
      ItemTypeInfo* item_type_info = m_item_type_mng->typeFromId(cell_type);
      Int32 type_dimension = entity_info.m_dim;
      ofile << "\n";
      const ArcaneToMshTypeInfo& atm_type_info = arcaneToMshTypeInfo(cell_type);
      ConstArrayView<Int16> reorder_infos = atm_type_info.m_reorder_infos;
      ofile << type_dimension << " " << entity_info.m_entity_tag << " " << atm_type_info.m_msh_type
            << " " << items_of_current_type.size() << "\n";
      info() << "Writing items family=" << item_family->name() << " type=" << item_type_info->typeName()
             << " n=" << items_of_current_type.size()
             << " dimension=" << type_dimension;
      Int32 nb_node_for_type = item_type_info->nbLocalNode();
      ENUMERATE_ (ItemWithNodes, iitem, item_family->view(items_of_current_type)) {
        ItemWithNodes item = *iitem;
        ofile << item.uniqueId();
        // Traite l'éventuelle permutation entre numérotation MSH et Arcane
        if (!reorder_infos.empty()) {
          for (Int32 i = 0; i < nb_node_for_type; ++i)
            ofile << " " << item.node(reorder_infos[i]).uniqueId();
        }
        else {
          for (Int32 i = 0; i < nb_node_for_type; ++i)
            ofile << " " << item.node(i).uniqueId();
        }
        ofile << "\n";
      }
    }
  }
  ofile << "$EndElements\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshMeshWriter::
_writePeriodic(std::ostream& ofile)
{
  IItemFamily* node_family = m_mesh->nodeFamily();

  //   $Periodic
  //     numPeriodicLinks(size_t)
  //     entityDim(int) entityTag(int) entityTagMaster(int)
  //     numAffine(size_t) value(double) ...
  //     numCorrespondingNodes(size_t)
  //       nodeTag(size_t) nodeTagMaster(size_t)
  //       ...
  //     ...
  //   $EndPeriodic

  // Les informations de périodicité sont conservées dans \a m_mesh_info
  // qui n'existe pas si on n'est pas issu d'un maillage MSH.
  if (!m_has_periodic_info)
    return;
  ARCANE_CHECK_POINTER(m_mesh_info);

  ConstArrayView<MshPeriodicOneInfo> periodic_one_infos = m_mesh_info->m_periodic_info.m_periodic_list;
  Int32 nb_periodic = periodic_one_infos.size();
  ofile << "$Periodic\n";
  ofile << nb_periodic << "\n";

  UniqueArray<Int64> corresponding_nodes;
  UniqueArray<Int32> node_local_ids;
  ;
  // Sauve chaque lien de périodicité.
  for (const MshPeriodicOneInfo& one_info : periodic_one_infos) {
    // On ne sauve pas les entités associées au lien, donc on considère
    // que l'entité est de dimension zéro et les tags valent aussi zéro.
    ofile << "\n";
    ofile << "0 1 1\n";

    // Sauve les valeurs affines associées
    ConstArrayView<double> affine_values = one_info.m_affine_values;
    Int32 nb_affine = affine_values.size();
    ofile << nb_affine;
    for (Int32 i = 0; i < nb_affine; ++i)
      ofile << " " << affine_values[i];
    ofile << "\n";

    // Sauve les couples de noeuds (esclave/maitre)
    // On ne sauve que les couples dont au moins l'un de des deux noeuds
    // est présent dans notre sous-domaine.
    Int32 nb_orig_node = one_info.m_nb_corresponding_node;
    ConstArrayView<Int64> orig_corresponding_nodes = one_info.m_corresponding_nodes;
    node_local_ids.resize(nb_orig_node * 2);
    node_family->itemsUniqueIdToLocalId(node_local_ids, orig_corresponding_nodes, false);
    corresponding_nodes.reserve(nb_orig_node * 2);
    corresponding_nodes.clear();
    for (Int32 i = 0; i < nb_orig_node; ++i) {
      Int32 slave_index = (i * 2);
      Int32 master_index = slave_index + 1;
      bool has_slave = node_local_ids[slave_index] != NULL_ITEM_LOCAL_ID;
      bool has_master = node_local_ids[master_index] != NULL_ITEM_LOCAL_ID;
      if (has_slave || has_master) {
        corresponding_nodes.add(orig_corresponding_nodes[slave_index]);
        corresponding_nodes.add(orig_corresponding_nodes[master_index]);
      }
    }
    Int32 nb_new_node = corresponding_nodes.size() / 2;
    ofile << nb_new_node << "\n";
    for (Int32 i = 0; i < nb_new_node; ++i)
      ofile << corresponding_nodes[(i * 2)] << " " << corresponding_nodes[(i * 2) + 1] << "\n";
  }

  ofile << "$EndPeriodic\n";
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

void MshMeshWriter::
_initTypes()
{
  m_arcane_to_msh_type_infos.resize(NB_BASIC_ITEM_TYPE);
  // Initialise les types.
  // Il faut le faire au début de la lecture et ne plus en ajouter après.
  _addArcaneTypeInfo(ITI_Vertex, MSH_PNT);
  _addArcaneTypeInfo(ITI_Line2, MSH_LIN_2);
  _addArcaneTypeInfo(ITI_Cell3D_Line2, MSH_LIN_2);
  _addArcaneTypeInfo(ITI_Triangle3, MSH_TRI_3);
  _addArcaneTypeInfo(ITI_Cell3D_Triangle3, MSH_TRI_3);
  _addArcaneTypeInfo(ITI_Quad4, MSH_QUA_4);
  _addArcaneTypeInfo(ITI_Cell3D_Quad4, MSH_QUA_4);
  _addArcaneTypeInfo(ITI_Tetraedron4, MSH_TET_4);
  _addArcaneTypeInfo(ITI_Hexaedron8, MSH_HEX_8);
  _addArcaneTypeInfo(ITI_Pentaedron6, MSH_PRI_6);
  _addArcaneTypeInfo(ITI_Pyramid5, MSH_PYR_5);
  _addArcaneTypeInfo(ITI_Triangle6, MSH_TRI_6);
  {
    FixedArray<Int16, 10> x({ 0, 1, 2, 3, 4, 5, 6, 7, 9, 8 });
    _addArcaneTypeInfo(ITI_Tetraedron10, MSH_TET_10, x.view());
  }
  {
    FixedArray<Int16, 20> x({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 9, 17, 10, 18, 19, 12, 15, 13, 14 });
    _addArcaneTypeInfo(ITI_Hexaedron20, MSH_HEX_20, x.view());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshMeshWriter::
_addArcaneTypeInfo(ItemTypeId arcane_type, Int32 msh_type, ConstArrayView<Int16> reorder_infos)
{
  if (arcane_type.isNull())
    ARCANE_FATAL("Null Arcane Type {0}", arcane_type);
  if (arcane_type >= m_arcane_to_msh_type_infos.size())
    ARCANE_FATAL("Invalid Arcane type '{0}'", arcane_type);
  m_arcane_to_msh_type_infos[arcane_type] = ArcaneToMshTypeInfo(arcane_type, msh_type, reorder_infos);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const MshMeshWriter::ArcaneToMshTypeInfo& MshMeshWriter::
arcaneToMshTypeInfo(ItemTypeId arcane_type) const
{
  if (arcane_type < m_arcane_to_msh_type_infos.size()) {
    const ArcaneToMshTypeInfo& tx = m_arcane_to_msh_type_infos[arcane_type];
    if (tx.m_msh_type > 0)
      return tx;
  }
  ARCANE_THROW(NotSupportedException, "Arcane type '{0}' is not supported in MSH writer", arcane_type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Écriture des fichiers de maillage au format msh.
 */
class MshMeshWriterService
: public AbstractService
, public IMeshWriter
{
 public:

  explicit MshMeshWriterService(const ServiceBuildInfo& sbi);

 public:

  void build() override {}
  bool writeMeshToFile(IMesh* mesh, const String& file_name) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MshMeshWriterService::
MshMeshWriterService(const ServiceBuildInfo& sbi)
: AbstractService(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MshMeshWriterService::
writeMeshToFile(IMesh* mesh, const String& file_name)
{
  MshMeshWriter writer(mesh);
  writer.writeMesh(file_name);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Obsolète. Utiliser 'MshMeshReader' à la place
ARCANE_REGISTER_SERVICE(MshMeshWriterService,
                        ServiceProperty("MshNewMeshWriter", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshWriter));

ARCANE_REGISTER_SERVICE(MshMeshWriterService,
                        ServiceProperty("MshMeshWriter", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshWriter));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
