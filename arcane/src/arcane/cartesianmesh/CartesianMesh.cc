﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMesh.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Maillage cartésien.                                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/AutoDestroyUserData.h"
#include "arcane/utils/IUserDataList.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Event.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/Properties.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/MeshStats.h"
#include "arcane/core/ICartesianMeshGenerationInfo.h"
#include "arcane/core/MeshEvents.h"
#include "arcane/core/MeshKind.h"
#include "arcane/core/internal/IMeshInternal.h"

#include "arcane/cartesianmesh/CartesianPatchGroup.h"
#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CartesianConnectivity.h"
#include "arcane/cartesianmesh/CartesianMeshRenumberingInfo.h"
#include "arcane/cartesianmesh/CartesianMeshCoarsening.h"
#include "arcane/cartesianmesh/CartesianMeshCoarsening2.h"
#include "arcane/cartesianmesh/CartesianMeshPatchListView.h"
#include "arcane/cartesianmesh/internal/CartesianMeshPatch.h"
#include "arcane/cartesianmesh/internal/ICartesianMeshInternal.h"

#include "arcane/cartesianmesh/internal/CartesianMeshUniqueIdRenumbering.h"
#include "arcane/cartesianmesh/v2/CartesianMeshUniqueIdRenumberingV2.h"
#include "arcane/cartesianmesh/CartesianMeshNumberingMng.h"

#include "arcane/cartesianmesh/CartesianMeshAMRPatchMng.h"
#include "arcane/core/IGhostLayerMng.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \defgroup ArcaneCartesianMesh Maillages cartésiens.
 *
 * Ensemble des classes assurant la gestion des maillage cartésiens.
 *
 * Pour plus de renseignements, se reporter à la page \ref arcanedoc_entities_cartesianmesh.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Infos spécifiques à un maillage cartésien.
 */
class CartesianMeshImpl
: public TraceAccessor
, public ICartesianMesh
{
  class InternalApi
  : public ICartesianMeshInternal
  {
   public:

    explicit InternalApi(CartesianMeshImpl* cartesian_mesh)
    : m_cartesian_mesh(cartesian_mesh)
    {
    }

   public:

    Ref<CartesianMeshCoarsening2> createCartesianMeshCoarsening2() override
    {
      return m_cartesian_mesh->_createCartesianMeshCoarsening2();
    }
    void addPatchFromExistingChildren(ConstArrayView<Int32> parent_cells_local_id) override
    {
      m_cartesian_mesh->_addPatchFromExistingChildren(parent_cells_local_id);
    }
    void initCartesianMeshAMRPatchMng() override
    {
      if (m_numbering_mng.isNull()) {
        initCartesianMeshNumberingMng();
      }
      if (m_amr_mng.isNull()) {
        m_amr_mng = makeRef(new CartesianMeshAMRPatchMng(m_cartesian_mesh, m_numbering_mng.get()));
      }
    }
    Ref<ICartesianMeshAMRPatchMng> cartesianMeshAMRPatchMng() override
    {
      return m_amr_mng;
    }
    void initCartesianMeshNumberingMng() override
    {
      if (m_numbering_mng.isNull()) {
        m_numbering_mng = makeRef(new CartesianMeshNumberingMng(m_cartesian_mesh->mesh()));
      }
    }
    Ref<ICartesianMeshNumberingMng> cartesianMeshNumberingMng() override
    {
      return m_numbering_mng;
    }

   private:

    CartesianMeshImpl* m_cartesian_mesh = nullptr;
    Ref<ICartesianMeshAMRPatchMng> m_amr_mng;
    Ref<ICartesianMeshNumberingMng> m_numbering_mng;
  };

 public:

  explicit CartesianMeshImpl(IMesh* mesh);

 public:

  void build() override;

  //! Maillage associé à ce maillage cartésien
  IMesh* mesh() const override { return m_mesh; }

  //! Gestionnaire de trace associé.
  ITraceMng* traceMng() const override { return TraceAccessor::traceMng(); }

  CellDirectionMng cellDirection(eMeshDirection dir) override
  {
    return m_all_items_direction_info->cellDirection(dir);
  }

  CellDirectionMng cellDirection(Integer idir) override
  {
    return m_all_items_direction_info->cellDirection(idir);
  }

  FaceDirectionMng faceDirection(eMeshDirection dir) override
  {
    return m_all_items_direction_info->faceDirection(dir);
  }

  FaceDirectionMng faceDirection(Integer idir) override
  {
    return m_all_items_direction_info->faceDirection(idir);
  }

  NodeDirectionMng nodeDirection(eMeshDirection dir) override
  {
    return m_all_items_direction_info->nodeDirection(dir);
  }

  NodeDirectionMng nodeDirection(Integer idir) override
  {
    return m_all_items_direction_info->nodeDirection(idir);
  }

  void computeDirections() override;

  void recreateFromDump() override;

  CartesianConnectivity connectivity() override
  {
    return m_connectivity;
  }

  Int32 nbPatch() const override { return m_patch_group.nbPatch(); }
  ICartesianMeshPatch* patch(Int32 index) const override { return m_patch_group.patch(index).get(); }
  CartesianPatch amrPatch(Int32 index) const override { return CartesianPatch(m_patch_group.patch(index).get()); }
  CartesianMeshPatchListView patches() const override { return m_patch_group.patchListView(); }

  void refinePatch2D(Real2 position,Real2 length) override;
  void refinePatch3D(Real3 position,Real3 length) override;
  void refinePatch(const AMRZonePosition& position) override;

  void coarseZone2D(Real2 position, Real2 length) override;
  void coarseZone3D(Real3 position, Real3 length) override;
  void coarseZone(const AMRZonePosition& position) override;

  Integer reduceNbGhostLayers(Integer level, Integer target_nb_ghost_layers) override;

  void renumberItemsUniqueId(const CartesianMeshRenumberingInfo& v) override;

  void checkValid() const override;

  Ref<CartesianMeshCoarsening> createCartesianMeshCoarsening() override;

  //! API interne à Arcane
  ICartesianMeshInternal* _internalApi() override { return &m_internal_api; }

 private:

  // Implémentation de 'ICartesianMeshInternal'
  Ref<CartesianMeshCoarsening2> _createCartesianMeshCoarsening2();
  void _addPatchFromExistingChildren(ConstArrayView<Int32> parent_cells_local_id);

 private:

  InternalApi m_internal_api;
  //! Indice dans la numérotation locale de la maille, de la face dans
  // la direction X, Y ou Z
  Int32 m_local_face_direction[3] = { -1, -1, -1 };
  IMesh* m_mesh = nullptr;
  Ref<CartesianMeshPatch> m_all_items_direction_info;
  CartesianConnectivity m_connectivity;
  UniqueArray<CartesianConnectivity::Index> m_nodes_to_cell_storage;
  UniqueArray<CartesianConnectivity::Index> m_cells_to_node_storage;
  UniqueArray<CartesianConnectivity::Permutation> m_permutation_storage;
  bool m_is_amr = false;
  //! Groupe de mailles pour chaque patch AMR.
  CartesianPatchGroup m_patch_group;
  ScopedPtrT<Properties> m_properties;

  EventObserverPool m_event_pool;
  bool m_is_mesh_event_added = false;
  Int64 m_mesh_timestamp = 0;
  eMeshAMRKind m_amr_type;

 private:

  void _computeMeshDirection(CartesianMeshPatch& cdi,eMeshDirection dir,
                             VariableCellReal3& cells_center,
                             VariableFaceReal3& faces_center,CellGroup all_cells,
                             NodeGroup all_nodes);
  void _applyRefine(const AMRZonePosition &position);
  void _removeCellsInPatches(ConstArrayView<Int32> const_array_view);
  void _applyCoarse(const AMRZonePosition &position);
  void _addPatch(const CellGroup& parent_cells);
  void _saveInfosInProperties();

  std::tuple<CellGroup, NodeGroup>
  _buildPatchGroups(const CellGroup& cells, Integer patch_level);
  void _checkNeedComputeDirections();
  void _checkAddObservableMeshChanged();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ICartesianMesh*
arcaneCreateCartesianMesh(IMesh* mesh)
{
  auto* cm = new CartesianMeshImpl(mesh);
  cm->build();
  return cm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshImpl::
CartesianMeshImpl(IMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_internal_api(this)
, m_mesh(mesh)
, m_nodes_to_cell_storage(platform::getDefaultDataAllocator())
, m_cells_to_node_storage(platform::getDefaultDataAllocator())
, m_permutation_storage(platform::getDefaultDataAllocator())
, m_patch_group(this)
, m_amr_type(mesh->meshKind().meshAMRKind())
{
  if (m_amr_type == eMeshAMRKind::PatchCartesianMeshOnly) {
    m_internal_api.initCartesianMeshNumberingMng();
    m_internal_api.initCartesianMeshAMRPatchMng();
  }
  m_all_items_direction_info = m_patch_group.groundPatch();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
build()
{
  m_properties = new Properties(*(mesh()->properties()),"CartesianMesh");
}

namespace
{
const Int32 SERIALIZE_VERSION = 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
_checkNeedComputeDirections()
{
  Int64 new_timestamp = mesh()->timestamp();
  if (m_mesh_timestamp!=new_timestamp){
    info() << "Mesh timestamp has changed (old=" << m_mesh_timestamp << " new=" << new_timestamp << ")";
    computeDirections();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
_saveInfosInProperties()
{
  // Sauve le numéro de version pour être sur que c'est OK en reprise
  m_properties->set("Version",SERIALIZE_VERSION);

  // Sauve les informations des patches
  UniqueArray<String> patch_group_names;
  for (Integer i = 1; i < m_patch_group.nbPatch(); ++i) {
    patch_group_names.add(m_patch_group.cells(i).name());
  }
  m_properties->set("PatchGroupNames",patch_group_names);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
recreateFromDump()
{
  info() << "Creating 'CartesianMesh' infos from dump";

  // Sauve le numéro de version pour être sur que c'est OK en reprise
  Int32 v = m_properties->getInt32("Version");
  if (v!=SERIALIZE_VERSION)
    ARCANE_FATAL("Bad serializer version: trying to read from incompatible checkpoint v={0} expected={1}",
                 v,SERIALIZE_VERSION);

  // Récupère les noms des groupes des patchs
  UniqueArray<String> patch_group_names;
  m_properties->get("PatchGroupNames",patch_group_names);
  info(4) << "Found n=" << patch_group_names.size() << " patchs";
  m_patch_group.clear();
  m_all_items_direction_info = m_patch_group.groundPatch();
  IItemFamily* cell_family = m_mesh->cellFamily();
  for( const String& x : patch_group_names ){
    CellGroup group = cell_family->findGroup(x);
    if (group.null())
      ARCANE_FATAL("Can not find cell group '{0}'",x);
    m_patch_group.addPatch(group);
  }

  computeDirections();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
_checkAddObservableMeshChanged()
{
  if (m_is_mesh_event_added)
    return;
  m_is_mesh_event_added = true;
  // Pour appeler automatiquement 'computeDirections()' après un appel à
  // IMesh::prepareForDump().
  auto f1 = [&](const MeshEventArgs&){ this->_checkNeedComputeDirections(); };
  mesh()->eventObservable(eMeshEventType::EndPrepareDump).attach(m_event_pool,f1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
computeDirections()
{
  info() << "CartesianMesh: computeDirections()";

  m_mesh_timestamp = mesh()->timestamp();
  _checkAddObservableMeshChanged();

  m_is_amr = m_mesh->isAmrActivated();

  VariableCellReal3 cells_center(VariableBuildInfo(m_mesh,"TemporaryCartesianMeshCellCenter"));
  VariableFaceReal3 faces_center(VariableBuildInfo(m_mesh,"TemporaryCartesianMeshFaceCenter"));

  // Calcule les coordonnées du centre des mailles.
  VariableNodeReal3& nodes_coord = m_mesh->nodesCoordinates();
  ENUMERATE_CELL(icell,m_mesh->allCells()){
    Cell cell = *icell;
    Real3 center;
    for( NodeLocalId inode : cell.nodeIds() )
      center += nodes_coord[inode];
    center /= cell.nbNode();
    cells_center[icell] = center;
  }
  ENUMERATE_FACE(iface,m_mesh->allFaces()){
    Face face = *iface;
    Real3 center;
    for( NodeLocalId inode : face.nodeIds() )
      center += nodes_coord[inode];
    center /= face.nbNode();
    faces_center[iface] = center;
  }

  IItemFamily* cell_family = m_mesh->cellFamily();
  IItemFamily* node_family = m_mesh->nodeFamily();
  Int32 next_face_x = -1;
  Int32 next_face_y = -1;
  Int32 next_face_z = -1;

  CellVectorView cell_view = cell_family->allItems().view();
  Cell cell0 = cell_view[0];
  Integer nb_face = cell0.nbFace();
  Integer nb_node = cell0.nbNode();
  Real3 cell_center = cells_center[cell0];

  info(4) << "sizeof(CellDirectionMng)=" << sizeof(CellDirectionMng)
         << " sizeof(FaceDirectionMng)=" << sizeof(FaceDirectionMng)
         << " sizeof(NodelDirectionMng)=" << sizeof(NodeDirectionMng);
  info(4) << "sizeof(IndexedItemConnectivityViewBase)=" << sizeof(IndexedItemConnectivityViewBase)
         << " sizeof(CellInfoListView)=" << sizeof(CellInfoListView);
  info(4) << "Cartesian mesh compute directions is_amr=" << m_is_amr;

  for( Integer i=0; i<nb_node; ++i ){
    Node node = cell0.node(i);
    info(4) << "Node I=" << i << " node=" << ItemPrinter(node) << " pos=" << nodes_coord[node];
  }

  bool is_3d = m_mesh->dimension() == 3;

  // On suppose que toutes les mailles ont le même sens de numérotation dans le maillage.
  // Par exemple, pour toutes les mailles, la face d'indice 0 est celle du haut, celle
  // d'indice 1 celle de droite.
  if (is_3d) {
    Real max_x = -1;
    Real max_y = -1;
    Real max_z = -1;

    for (Integer i = 0; i < nb_face; ++i) {
      Face f = cell0.face(i);

      Real3 next_center = faces_center[f];

      Real diff_x = next_center.x - cell_center.x;
      Real diff_y = next_center.y - cell_center.y;
      Real diff_z = next_center.z - cell_center.z;

      info(4) << "NEXT_FACE=" << ItemPrinter(f) << " center=" << next_center << " diff=" << Real3(diff_x, diff_y, diff_z);

      if (diff_x > max_x) {
        max_x = diff_x;
        next_face_x = i;
      }

      if (diff_y > max_y) {
        max_y = diff_y;
        next_face_y = i;
      }

      if (diff_z > max_z) {
        max_z = diff_z;
        next_face_z = i;
      }
    }
    info(4) << "Advance in direction X -> " << next_face_x;
    info(4) << "Advance in direction Y -> " << next_face_y;
    info(4) << "Advance in direction Z -> " << next_face_z;
  }
  else {
    Real max_x = -1;
    Real max_y = -1;

    for (Integer i = 0; i < nb_face; ++i) {
      Face f = cell0.face(i);

      Real3 next_center = faces_center[f];

      Real diff_x = next_center.x - cell_center.x;
      Real diff_y = next_center.y - cell_center.y;

      info(4) << "NEXT_FACE=" << ItemPrinter(f) << " center=" << next_center << " diff=" << Real2(diff_x, diff_y);

      if (diff_x > max_x) {
        max_x = diff_x;
        next_face_x = i;
      }

      if (diff_y > max_y) {
        max_y = diff_y;
        next_face_y = i;
      }
    }
    info(4) << "Advance in direction X -> " << next_face_x;
    info(4) << "Advance in direction Y -> " << next_face_y;
  }
  m_all_items_direction_info->_internalComputeNodeCellInformations(cell0,cells_center[cell0],nodes_coord);

  info() << "Informations from IMesh properties:";

  auto* cmgi = ICartesianMeshGenerationInfo::getReference(m_mesh,true);

  info() << "GlobalNbCell = " << cmgi->globalNbCells();
  info() << "OwnNbCell: " << cmgi->ownNbCells();
  info() << "SubDomainOffset: " << cmgi->subDomainOffsets();
  info() << "OwnCellOffset: " << cmgi->ownCellOffsets();

  CellGroup all_cells = cell_family->allItems();
  NodeGroup all_nodes = node_family->allItems();
  if (m_is_amr){
    auto x = _buildPatchGroups(mesh()->allLevelCells(0),0);
    all_cells = std::get<0>(x);
    all_nodes = std::get<1>(x);
  }

  if (next_face_x!=(-1)){
    m_local_face_direction[MD_DirX] = next_face_x;
    _computeMeshDirection(*m_all_items_direction_info.get(),MD_DirX,cells_center,faces_center,all_cells,all_nodes);
  }
  if (next_face_y!=(-1)){
    m_local_face_direction[MD_DirY] = next_face_y;
    _computeMeshDirection(*m_all_items_direction_info.get(),MD_DirY,cells_center,faces_center,all_cells,all_nodes);
  }
  if (next_face_z != (-1)) {
    m_local_face_direction[MD_DirZ] = next_face_z;
    _computeMeshDirection(*m_all_items_direction_info.get(),MD_DirZ,cells_center,faces_center,all_cells,all_nodes);
  }

  // Positionne les informations par direction
  for( Integer idir=0, nb_dir=mesh()->dimension(); idir<nb_dir; ++idir ){
    CellDirectionMng& cdm = m_all_items_direction_info->cellDirection(idir);
    cdm._internalSetOffsetAndNbCellInfos(cmgi->globalNbCells()[idir], cmgi->ownNbCells()[idir],
                                         cmgi->subDomainOffsets()[idir], cmgi->ownCellOffsets()[idir]);
  }

  info() << "Compute cartesian connectivity";

  m_permutation_storage.resize(1);
  m_permutation_storage[0].compute();
  m_nodes_to_cell_storage.resize(mesh()->nodeFamily()->maxLocalId());
  m_cells_to_node_storage.resize(mesh()->cellFamily()->maxLocalId());
  m_connectivity._setStorage(m_nodes_to_cell_storage,m_cells_to_node_storage,&m_permutation_storage[0]);
  m_connectivity._computeInfos(mesh(),nodes_coord,cells_center);

  // Ajoute informations de connectivités pour les patchs AMR
  // TODO: supporter plusieurs appels à cette méthode ?
  for (Integer patch_index = 1; patch_index < m_patch_group.nbPatch(); ++patch_index) {
    CellGroup cells = m_patch_group.cells(patch_index);
    Ref<CartesianMeshPatch> patch = m_patch_group.patch(patch_index);
    info() << "AMR Patch name=" << cells.name() << " size=" << cells.size() << " index=" << patch_index << " nbPatch=" << m_patch_group.nbPatch();
    patch->_internalComputeNodeCellInformations(cell0, cells_center[cell0], nodes_coord);
    auto [patch_cells, patch_nodes] = _buildPatchGroups(cells, patch_index);
    _computeMeshDirection(*patch.get(), MD_DirX, cells_center, faces_center, patch_cells, patch_nodes);
    _computeMeshDirection(*patch.get(), MD_DirY, cells_center, faces_center, patch_cells, patch_nodes);
    if (is_3d)
      _computeMeshDirection(*patch.get(), MD_DirZ, cells_center, faces_center, patch_cells, patch_nodes);
  }

  if (arcaneIsCheck())
    checkValid();

  _saveInfosInProperties();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::tuple<CellGroup,NodeGroup> CartesianMeshImpl::
_buildPatchGroups(const CellGroup& cells,Integer patch_level)
{
  // On créé un groupe pour chaque patch en garantissant que l'ordre de parcours
  // est celui des uniqueId() des entités
  // TODO: à terme, il faudrait que l'ordre de parcours soit le même que
  // celui du maillage cartésien. Pour cela, il faut soit que les uniqueId()
  // des mailles/noeuds créés soient dans le même ordre que le maillage cartésien,
  // soit que la fonction de tri soit spécifique à ce type de maillage.
  NodeGroup nodes = cells.nodeGroup();
  IItemFamily* cell_family = cells.itemFamily();
  IItemFamily* node_family = nodes.itemFamily();

  String cell_group_name = String("AMRPatchCells") + patch_level;
  CellGroup patch_cells = cell_family->createGroup(cell_group_name,Int32ConstArrayView(),true);
  // Met les mêmes mailles que \a cells mais force le tri
  patch_cells.setItems(cells.view().localIds(),true);

  String node_group_name = String("AMRPatchNodes") + patch_level;
  NodeGroup patch_nodes = node_family->createGroup(node_group_name,Int32ConstArrayView(),true);
  // Met les mêmes noeuds que \a nodes mais force le tri
  patch_nodes.setItems(nodes.view().localIds(),true);
  info(4) << "PATCH_CELLS name=" << patch_cells.name() << " size=" << patch_cells.size();
  info(4) << "PATCH_NODES name=" << patch_nodes.name() << " size=" << patch_nodes.size();
  return { patch_cells, patch_nodes };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
_computeMeshDirection(CartesianMeshPatch& cdi,eMeshDirection dir,VariableCellReal3& cells_center,
                      VariableFaceReal3& faces_center,CellGroup all_cells,NodeGroup all_nodes)
{
  IItemFamily* cell_family = m_mesh->cellFamily();
  IItemFamily* face_family = m_mesh->faceFamily();
  IItemFamily* node_family = m_mesh->nodeFamily();

  Int32 max_cell_id = cell_family->maxLocalId();
  Int32 max_face_id = face_family->maxLocalId();
  Int32 max_node_id = node_family->maxLocalId();

  CellDirectionMng& cell_dm = cdi.cellDirection(dir);
  cell_dm._internalResizeInfos(max_cell_id);

  FaceDirectionMng& face_dm = cdi.faceDirection(dir);
  face_dm._internalResizeInfos(max_face_id);

  NodeDirectionMng& node_dm = cdi.nodeDirection(dir);
  node_dm._internalResizeInfos(max_node_id);

  //TODO: attention à remettre à jour après changement de maillage.
  info(4) << "COMPUTE DIRECTION dir=" << dir;

  Int32 prev_local_face = -1;
  Int32 next_local_face = m_local_face_direction[dir];
  Integer mesh_dim = m_mesh->dimension();
  // Calcul le numero local de face oppose à la face suivante.
  if (mesh_dim==2)
    prev_local_face = (next_local_face + 2) % 4;
  else if (mesh_dim==3)
    prev_local_face = (next_local_face + 3) % 6;

  cell_dm._internalSetLocalFaceIndex(next_local_face,prev_local_face);

  // Positionne pour chaque maille les faces avant et après dans la direction.
  // On s'assure que ces entités sont dans le groupe des entités de la direction correspondante
  std::set<Int32> cells_set;
  ENUMERATE_CELL(icell,all_cells){
    cells_set.insert(icell.itemLocalId());
  }

  // Calcule les mailles devant/derrière. En cas de patch AMR, il faut que ces deux mailles
  // soient de même niveau
  ENUMERATE_CELL(icell,all_cells){
    Cell cell = *icell;
    Int32 my_level = cell.level();
    Face next_face = cell.face(next_local_face);
    Cell next_cell = next_face.backCell()==cell ? next_face.frontCell() : next_face.backCell();
    if (cells_set.find(next_cell.localId())==cells_set.end())
      next_cell = Cell();
    else if (next_cell.level()!=my_level)
      next_cell = Cell();

    Face prev_face = cell.face(prev_local_face);
    Cell prev_cell = prev_face.backCell()==cell ? prev_face.frontCell() : prev_face.backCell();
    if (cells_set.find(prev_cell.localId())==cells_set.end())
      prev_cell = Cell();
    else if (prev_cell.level()!=my_level)
      prev_cell = Cell();
    cell_dm.m_infos_view[icell.itemLocalId()] = CellDirectionMng::ItemDirectionInfo(next_cell,prev_cell);
  }
  cell_dm._internalComputeInnerAndOuterItems(all_cells);
  face_dm._internalComputeInfos(cell_dm,cells_center,faces_center);
  node_dm._internalComputeInfos(cell_dm,all_nodes,cells_center);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
refinePatch2D(Real2 position,Real2 length)
{
  info() << "REFINEMENT 2D position=" << position << " length=" << length;
  refinePatch({ position, length });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
refinePatch3D(Real3 position, Real3 length)
{
  info() << "REFINEMENT 3D position=" << position << " length=" << length;
  refinePatch({ position, length });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
refinePatch(const AMRZonePosition& position)
{
  _applyRefine(position);
  _saveInfosInProperties();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
coarseZone2D(Real2 position, Real2 length)
{
  info() << "COARSEN 2D position=" << position << " length=" << length;
  coarseZone({ position, length });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
coarseZone3D(Real3 position, Real3 length)
{
  info() << "COARSEN 3D position=" << position << " length=" << length;
  coarseZone({ position, length });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
coarseZone(const AMRZonePosition& position)
{
  _applyCoarse(position);
  _saveInfosInProperties();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CartesianMeshImpl::
reduceNbGhostLayers(Integer level, Integer target_nb_ghost_layers)
{
  if (level < 1) {
    ARCANE_FATAL("You cannot reduce number of ghost layer of level 0 with this method");
  }

  // Nombre de couche de maille fantôme max. Bof; à modifier.
  const Int32 max_nb_layer = 128;
  Int32 level_max = 0;

  ENUMERATE_ (Cell, icell, m_mesh->allCells()) {
    level_max = std::max(level_max, icell->level());
  }

  level_max = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, level_max);
  //debug() << "Level max : " << level_max;

  computeDirections();

  Integer level_0_nb_ghost_layer = m_mesh->ghostLayerMng()->nbGhostLayer();
  //debug() << "NbGhostLayers level 0 : " << level_0_nb_ghost_layer;

  if (level_0_nb_ghost_layer == 0) {
    return 0;
  }

  Integer nb_ghost_layer = Convert::toInt32(level_0_nb_ghost_layer * pow(2, level));

  //debug() << "NbGhostLayers level " << level << " : " << nb_ghost_layer;

  // On considère qu'on a toujours 2*2 mailles filles (2*2*2 en 3D).
  if (target_nb_ghost_layers % 2 != 0) {
    target_nb_ghost_layers++;
  }

  if (target_nb_ghost_layers == nb_ghost_layer) {
    return nb_ghost_layer;
  }

  //debug() << "TargetNbGhostLayers level " << level << " : " << target_nb_ghost_layers;

  Integer parent_level = level - 1;
  Integer parent_target_nb_ghost_layer = target_nb_ghost_layers / 2;

  // TODO AH : On est forcé de dé-raffiner niveau par niveau. À changer.
  UniqueArray<UniqueArray<Int32>> cell_lid2(level_max);

  //UniqueArray<Int32> cell_lid;
  std::function<void(Cell)> children_list;

  children_list = [&cell_lid2, &children_list](Cell cell) -> void {
    for (Integer i = 0; i < cell.nbHChildren(); ++i) {
      //debug() << "child of lid=" << cell.localId() << " : lid=" << cell.hChild(i).localId() << " -- level : " << cell.level();
      cell_lid2[cell.level()].add(cell.hChild(i).localId());
      children_list(cell.hChild(i));
    }
  };

  // Algorithme de numérotation des couches de mailles fantômes.
  {
    VariableNodeInt32 level_node{ VariableBuildInfo{ m_mesh, "LevelNode" } };
    level_node.fill(-1);

    VariableCellInt32 level_cell{ VariableBuildInfo{ m_mesh, "LevelCell" } };
    level_cell.fill(-1);

    ENUMERATE_ (Face, iface, m_mesh->allFaces()) {
      Cell front_cell = iface->frontCell();
      Cell back_cell = iface->backCell();
      if (
      ((front_cell.null() || (!front_cell.isOwn() && front_cell.level() == parent_level)) && ((!back_cell.null()) && (back_cell.isOwn() && back_cell.level() == parent_level))) ||
      ((back_cell.null() || (!back_cell.isOwn() && back_cell.level() == parent_level)) && ((!front_cell.null()) && (front_cell.isOwn() && front_cell.level() == parent_level)))) {
        for (Node node : iface->nodes()) {
          level_node[node] = 0;
          //debug() << "Node layer 0 : " << node.uniqueId();
        }
      }
    }

    bool is_modif = true;
    Int32 current_layer = 0;
    while (is_modif) {
      is_modif = false;

      ENUMERATE_ (Cell, icell, m_mesh->allCells()) {
        if (icell->isOwn() || icell->level() != parent_level || level_cell[icell] != -1) {
          continue;
        }

        Int32 min = max_nb_layer;
        Int32 max = -1;

        for (Node node : icell->nodes()) {
          Int32 nlevel = level_node[node];
          if (nlevel != -1) {
            min = std::min(min, nlevel);
            max = std::max(max, nlevel);
          }
        }

        // On fait couche par couche (voir pour enlever cette limitation).
        if (min != current_layer) {
          continue;
        }

        // Maille n'ayant pas de nodes déjà traités.
        if (min == max_nb_layer && max == -1) {
          continue;
        }

        Integer new_level = ((min == max) ? min + 1 : max);

        for (Node node : icell->nodes()) {
          Int32 nlevel = level_node[node];
          if (nlevel == -1) {
            level_node[node] = new_level;
            //debug() << "Node layer " << new_level << " : " << node.uniqueId();
            is_modif = true;
          }
        }

        level_cell[icell] = min;

        //debug() << "Cell uid : " << icell->uniqueId()
        //        << " -- Layer : " << min;

        if (min >= parent_target_nb_ghost_layer) {
          children_list(*icell);
        }
      }
      current_layer++;
      if (current_layer >= max_nb_layer) {
        ARCANE_FATAL("Error in ghost layer counter algo. Report it plz.");
      }
    }
  }

  for (Integer i = level_max - 1; i >= 0; --i) {
    // Une comm pour en éviter plein d'autres.
    if (m_mesh->parallelMng()->reduce(Parallel::ReduceMax, cell_lid2[i].size()) == 0) {
      continue;
    }
    //debug() << "Removing children of ghost cell (parent level=" << i << ") (children localIds) : " << cell_lid2[i];

    m_mesh->modifier()->flagCellToCoarsen(cell_lid2[i]);
    m_mesh->modifier()->coarsenItemsV2(false);
  }

  info() << "Nb ghost layer for level " << level << " : " << target_nb_ghost_layers;

  return target_nb_ghost_layers;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
_addPatchFromExistingChildren(ConstArrayView<Int32> parent_cells_local_id)
{
  IItemFamily* cell_family = m_mesh->cellFamily();
  Integer index = m_patch_group.nextIndexForNewPatch();
  String parent_group_name = String("CartesianMeshPatchParentCells")+index;
  CellGroup parent_cells = cell_family->createGroup(parent_group_name,parent_cells_local_id,true);
  _addPatch(parent_cells);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé un patch avec tous les enfants du groupe \a parent_cells.
 */
void CartesianMeshImpl::
_addPatch(const CellGroup& parent_cells)
{
  Integer index = m_patch_group.nextIndexForNewPatch();
  info() << "Add patch index : " << index;
  // Créé le groupe contenant les mailles AMR
  // Il s'agit des mailles filles de \a parent_cells
  String children_group_name = String("CartesianMeshPatchCells")+index;
  UniqueArray<Int32> children_local_id;
  ENUMERATE_(Cell,icell,parent_cells){
    Cell c = *icell;
    for(Integer k=0; k<c.nbHChildren(); ++k ){
      Cell child = c.hChild(k);
      children_local_id.add(child.localId());
    }
  }
  IItemFamily* cell_family = m_mesh->cellFamily();
  CellGroup children_cells = cell_family->createGroup(children_group_name, children_local_id, true);
  m_patch_group.addPatch(children_cells);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
_removeCellsInPatches(ConstArrayView<Int32> const_array_view)
{
  if (m_amr_type == eMeshAMRKind::Cell) {
    m_patch_group.removeCellsInAllPatches(const_array_view);
    m_patch_group.applyPatchEdit(true);
  }
  else if (m_amr_type == eMeshAMRKind::PatchCartesianMeshOnly) {
    m_patch_group.removeCellsInAllPatches(const_array_view);
    // SharedArray<Integer> altered_patches;
    // m_patch_group.removeCellsInAllPatches(const_array_view, altered_patches);
    // info() << "altered_patches : " << altered_patches;
    // for (Integer index : altered_patches) {
    //   m_patch_group.repairPatch(index, _internalApi()->cartesianMeshNumberingMng().get());
    // }
    m_patch_group.applyPatchEdit(true);
  }
  else if (m_amr_type == eMeshAMRKind::Patch) {
    ARCANE_FATAL("General patch AMR is not implemented. Please use PatchCartesianMeshOnly (3)");
  }
  else {
    ARCANE_FATAL("AMR is not enabled");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
_applyRefine(const AMRZonePosition& position)
{
  SharedArray<Int32> cells_local_id;
  position.cellsInPatch(mesh(), cells_local_id);

  IItemFamily* cell_family = m_mesh->cellFamily();
  Integer nb_cell = cells_local_id.size();
  info(4) << "Local_NbCellToRefine = " << nb_cell;
  Integer index = m_patch_group.nextIndexForNewPatch();
  String parent_group_name = String("CartesianMeshPatchParentCells")+index;
  CellGroup parent_cells = cell_family->createGroup(parent_group_name,cells_local_id,true);

  IParallelMng* pm = m_mesh->parallelMng();
  Int64 total_nb_cell = pm->reduce(Parallel::ReduceSum,nb_cell);
  info(4) << "Global_NbCellToRefine = " << total_nb_cell;
  if (total_nb_cell==0)
    return;

  if(m_amr_type == eMeshAMRKind::Cell) {
    debug() << "Refine with modifier() (for all mesh types)";
    m_mesh->modifier()->flagCellToRefine(cells_local_id);
    m_mesh->modifier()->adapt();
  }
  else if(m_amr_type == eMeshAMRKind::PatchCartesianMeshOnly) {
    debug() << "Refine with specific refiner (for cartesian mesh only)";
    computeDirections();
    m_internal_api.cartesianMeshAMRPatchMng()->flagCellToRefine(cells_local_id, true);
    m_internal_api.cartesianMeshAMRPatchMng()->refine();
  }
  else if(m_amr_type == eMeshAMRKind::Patch) {
    ARCANE_FATAL("General patch AMR is not implemented. Please use PatchCartesianMeshOnly (3)");
  }
  else{
    ARCANE_FATAL("AMR is not enabled");
  }

  {
    MeshStats ms(traceMng(),m_mesh,m_mesh->parallelMng());
    ms.dumpStats();
  }
  _addPatch(parent_cells);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
_applyCoarse(const AMRZonePosition& position)
{
  SharedArray<Int32> cells_local_id;
  position.cellsInPatch(mesh(), cells_local_id);

  Integer nb_cell = cells_local_id.size();
  info(4) << "Local_NbCellToCoarsen = " << nb_cell;

  IParallelMng* pm = m_mesh->parallelMng();
  Int64 total_nb_cell = pm->reduce(Parallel::ReduceSum, nb_cell);
  info(4) << "Global_NbCellToCoarsen = " << total_nb_cell;
  if (total_nb_cell == 0)
    return;

  _removeCellsInPatches(cells_local_id);

  if (m_amr_type == eMeshAMRKind::Cell) {
    debug() << "Coarse with modifier() (for all mesh types)";
    m_mesh->modifier()->flagCellToCoarsen(cells_local_id);
    m_mesh->modifier()->coarsenItemsV2(true);
  }
  else if (m_amr_type == eMeshAMRKind::PatchCartesianMeshOnly) {
    debug() << "Coarsen with specific coarser (for cartesian mesh only)";
    computeDirections();
    m_internal_api.cartesianMeshAMRPatchMng()->flagCellToCoarsen(cells_local_id, true);
    m_internal_api.cartesianMeshAMRPatchMng()->coarsen(true);
  }
  else if (m_amr_type == eMeshAMRKind::Patch) {
    ARCANE_FATAL("General patch AMR is not implemented. Please use PatchCartesianMeshOnly (3)");
  }
  else {
    ARCANE_FATAL("AMR is not enabled");
  }

  {
    MeshStats ms(traceMng(), m_mesh, m_mesh->parallelMng());
    ms.dumpStats();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
checkValid() const
{
  info(4) << "Check valid CartesianMesh";
  Integer nb_patch = nbPatch();
  for( Integer i=0; i<nb_patch; ++i ){
    ICartesianMeshPatch* p = patch(i);
    p->checkValid();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
renumberItemsUniqueId(const CartesianMeshRenumberingInfo& v)
{
  auto* cmgi = ICartesianMeshGenerationInfo::getReference(m_mesh,true);

  // Regarde d'abord si on renumérote les faces
  Int32 face_method = v.renumberFaceMethod();
  if (face_method!=0 && face_method!=1)
    ARCANE_FATAL("Invalid value '{0}' for renumberFaceMethod(). Valid values are 0 or 1",
                 face_method);
  if (face_method==1)
    ARCANE_THROW(NotImplementedException,"Method 1 for face renumbering");

  // Regarde ensuite les patchs si demandé.
  Int32 patch_method = v.renumberPatchMethod();
  if (patch_method < 0 || patch_method > 4)
    ARCANE_FATAL("Invalid value '{0}' for renumberPatchMethod(). Valid values are 0, 1, 2, 3 or 4",
                 patch_method);
    
  else if (patch_method == 1 || patch_method == 3 || patch_method == 4){
    CartesianMeshUniqueIdRenumbering renumberer(this,cmgi,v.parentPatch(),patch_method);
    renumberer.renumber();
  }
  else if (patch_method == 2){
    warning() << "The patch method 2 is experimental!";
    CartesianMeshUniqueIdRenumberingV2 renumberer(this,cmgi);
    renumberer.renumber();
  }

  // Termine par un tri éventuel.
  if (v.isSortAfterRenumbering()){
    info() << "Compacting and Sorting after renumbering";
    m_mesh->nodeFamily()->compactItems(true);
    m_mesh->faceFamily()->compactItems(true);
    m_mesh->cellFamily()->compactItems(true);
    computeDirections();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<CartesianMeshCoarsening> CartesianMeshImpl::
createCartesianMeshCoarsening()
{
  return makeRef(new CartesianMeshCoarsening(this));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<CartesianMeshCoarsening2> CartesianMeshImpl::
_createCartesianMeshCoarsening2()
{
  return makeRef(new CartesianMeshCoarsening2(this));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICartesianMesh* ICartesianMesh::
getReference(const MeshHandleOrMesh& mesh_handle_or_mesh,bool create)
{
  MeshHandle h = mesh_handle_or_mesh.handle();
  //TODO: faire lock pour multi-thread
  const char* name = "CartesianMesh";
  IUserDataList* udlist = h.meshUserDataList();

  IUserData* ud = udlist->data(name,true);
  if (!ud){
    if (!create)
      return nullptr;
    IMesh* mesh = h.meshOrNull();
    if (!mesh)
      ARCANE_FATAL("The mesh {0} is not yet created",h.meshName());
    ICartesianMesh* cm = arcaneCreateCartesianMesh(mesh);
    udlist->setData(name,new AutoDestroyUserData<ICartesianMesh>(cm));

    // Indique que le maillage est cartésien
    MeshKind mk = mesh->meshKind();
    mk.setMeshStructure(eMeshStructure::Cartesian);
    mesh->_internalApi()->setMeshKind(mk);

    return cm;
  }
  AutoDestroyUserData<ICartesianMesh>* adud = dynamic_cast<AutoDestroyUserData<ICartesianMesh>*>(ud);
  if (!adud)
    ARCANE_FATAL("Can not cast to ICartesianMesh*");
  return adud->data();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
