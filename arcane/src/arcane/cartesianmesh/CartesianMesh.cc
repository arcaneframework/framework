﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMesh.cc                                            (C) 2000-2024 */
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

#include "arcane/cartesianmesh/CartesianMeshAMRPatchMng.h"

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
      m_amr_mng = makeRef(new CartesianMeshAMRPatchMng(m_cartesian_mesh));
    }

    Ref<ICartesianMeshAMRPatchMng> cartesianMeshAMRPatchMng() override
    {
      return m_amr_mng;
    }

   private:

    CartesianMeshImpl* m_cartesian_mesh = nullptr;
    Ref<ICartesianMeshAMRPatchMng> m_amr_mng;
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

  Int32 nbPatch() const override { return m_amr_patches.size(); }
  ICartesianMeshPatch* patch(Int32 index) const override { return m_amr_patches[index].get(); }
  CartesianPatch amrPatch(Int32 index) const override { return CartesianPatch(m_amr_patches[index].get()); }
  CartesianMeshPatchListView patches() const override { return CartesianMeshPatchListView(m_amr_patches_pointer); }

  void refinePatch2D(Real2 position,Real2 length) override;
  void refinePatch3D(Real3 position,Real3 length) override;

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
  Int32 m_local_face_direction[3];
  IMesh* m_mesh = nullptr;
  Ref<CartesianMeshPatch> m_all_items_direction_info;
  CartesianConnectivity m_connectivity;
  UniqueArray<CartesianConnectivity::Index> m_nodes_to_cell_storage;
  UniqueArray<CartesianConnectivity::Index> m_cells_to_node_storage;
  UniqueArray<CartesianConnectivity::Permutation> m_permutation_storage;
  bool m_is_amr = false;
  //! Groupe de mailles parentes pour chaque patch AMR.
  UniqueArray<CellGroup> m_amr_patch_cell_groups;
  UniqueArray<Ref<CartesianMeshPatch>> m_amr_patches;
  UniqueArray<ICartesianMeshPatch*> m_amr_patches_pointer;
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
  void _applyRefine(ConstArrayView<Int32> cells_local_id);
  void _addPatch(const CellGroup& parent_group);
  void _saveInfosInProperties();

  std::tuple<CellGroup,NodeGroup>
  _buildPatchGroups(const CellGroup& cells,Integer patch_level);
  void _refinePatch(Real3 position,Real3 length,bool is_3d);
  void _checkNeedComputeDirections();
  void _checkAddObservableMeshChanged();
  void _addPatchInstance(const Ref<CartesianMeshPatch>& v)
  {
    m_amr_patches.add(v);
    m_amr_patches_pointer.add(v.get());
  }
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
, m_amr_type(mesh->meshKind().meshAMRKind())
{
  if (m_amr_type == eMeshAMRKind::PatchCartesianMeshOnly)
    m_internal_api.initCartesianMeshAMRPatchMng();

  m_all_items_direction_info = makeRef(new CartesianMeshPatch(this,-1));
  _addPatchInstance(m_all_items_direction_info);
  Integer nb_dir = mesh->dimension();
  for( Integer i=0; i<nb_dir; ++i ){
    m_local_face_direction[i] = -1;
  }
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
  for( const CellGroup& x : m_amr_patch_cell_groups ){
    patch_group_names.add(x.name());
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
  m_amr_patch_cell_groups.clear();
  IItemFamily* cell_family = m_mesh->cellFamily();
  for( const String& x : patch_group_names ){
    CellGroup group = cell_family->findGroup(x);
    if (group.null())
      ARCANE_FATAL("Can not find cell group '{0}'",x);
    m_amr_patch_cell_groups.add(group);
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

  m_amr_patches.clear();
  m_amr_patches_pointer.clear();
  _addPatchInstance(m_all_items_direction_info);

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

  // On suppose que toutes les mailles ont le même sens de numérotation dans le maillage.
  // Par exemple, pour toutes les mailles, la face d'indice 0 est celle du haut, celle
  // d'indice 1 celle de droite.
  for( Integer i=0; i<nb_face; ++i ){
    Face f = cell0.face(i);
    if (f.isSubDomainBoundary())
      continue;
    Cell next_cell = (f.backCell()==cell0) ? f.frontCell() : f.backCell();
    Real3 next_center = cells_center[next_cell];
    info(4) << "NEXT_CELL=" << ItemPrinter(next_cell) << " center=" << next_center \
            << " back=" << f.backCell().uniqueId()
            << " front=" << f.frontCell().uniqueId();

    Real diff_x = math::abs(next_center.x - cell_center.x);
    Real diff_y = math::abs(next_center.y - cell_center.y);
    Real diff_z = math::abs(next_center.z - cell_center.z);
    info(4) << "NEXT_CELL=" << ItemPrinter(next_cell) << " diff=" << Real3(diff_x,diff_y,diff_z);
    //TODO: Verifier qu'il s'agit bien de la maille apres et pas avant.
    // (tenir compte du signe de diff)
    if (diff_x>diff_y && diff_x>diff_z){
      // INC X
      next_face_x = i;
      info(4) << "Advance in direction X -> " << next_face_x;
    }
    else if (diff_y>diff_x && diff_y>diff_z){
      // INC Y
      next_face_y = i;
      info(4) << "Advance in direction Y -> " << next_face_y;
    }
    else if (diff_z>diff_x && diff_z>diff_y){
      // INC Z
      next_face_z = i;
      info(4) << "Advance in direction Z -> " << next_face_z;
    }
    else
      ARCANE_FATAL("Bad value for next cell");
  }

  bool is_3d = m_mesh->dimension()==3;
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
  if (next_face_z!=(-1)){
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
  // TODO: supporter plusieurs appels à cette méthode
  for( const CellGroup& cells : m_amr_patch_cell_groups ){
    Integer patch_index = m_amr_patches.size();
    info() << "AMR Patch name=" << cells.name() << " size=" << cells.size() << " index=" << patch_index;
    auto* cdi = new CartesianMeshPatch(this,patch_index);
    _addPatchInstance(makeRef(cdi));
    cdi->_internalComputeNodeCellInformations(cell0,cells_center[cell0],nodes_coord);
    auto [ patch_cells, patch_nodes ] = _buildPatchGroups(cells,patch_index);
    _computeMeshDirection(*cdi,MD_DirX,cells_center,faces_center,patch_cells,patch_nodes);
    _computeMeshDirection(*cdi,MD_DirY,cells_center,faces_center,patch_cells,patch_nodes);
    if (is_3d)
      _computeMeshDirection(*cdi,MD_DirZ,cells_center,faces_center,patch_cells,patch_nodes);
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
_refinePatch(Real3 position,Real3 length,bool is_3d)
{
  VariableNodeReal3& nodes_coord = m_mesh->nodesCoordinates();
  UniqueArray<Int32> cells_local_id;
  // Parcours les mailles actives et ajoute dans la liste des mailles
  // à raffiner celles qui sont contenues dans le boîte englobante
  // spécifiée dans le jeu de données.
  Real3 min_pos = position;
  Real3 max_pos = min_pos + length;
  cells_local_id.clear();
  ENUMERATE_CELL(icell,m_mesh->allActiveCells()){
    Cell cell = *icell;
    Real3 center;
    for( NodeLocalId inode : cell.nodeIds() )
      center += nodes_coord[inode];
    center /= cell.nbNode();
    bool is_inside_x = center.x>min_pos.x && center.x<max_pos.x;
    bool is_inside_y = center.y>min_pos.y && center.y<max_pos.y;
    bool is_inside_z = (center.z>min_pos.z && center.z<max_pos.z) || !is_3d;
    if (is_inside_x && is_inside_y && is_inside_z)
      cells_local_id.add(icell.itemLocalId());
  }
  _applyRefine(cells_local_id);
  _saveInfosInProperties();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
refinePatch2D(Real2 position,Real2 length)
{
  info() << "REFINEMENT 2D position=" << position << " length=" << length;
  Real3 position_3d(position.x,position.y,0.0);
  Real3 length_3d(length.x,length.y,0.0);
  _refinePatch(position_3d,length_3d,false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
refinePatch3D(Real3 position,Real3 length)
{
  info() << "REFINEMENT 3D position=" << position << " length=" << length;
  _refinePatch(position,length,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
_addPatchFromExistingChildren(ConstArrayView<Int32> parent_cells_local_id)
{
  IItemFamily* cell_family = m_mesh->cellFamily();
  Integer index = m_amr_patch_cell_groups.size();
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
  Integer index = m_amr_patch_cell_groups.size();
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
  CellGroup children_cells = cell_family->createGroup(children_group_name,children_local_id,true);
  m_amr_patch_cell_groups.add(children_cells);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshImpl::
_applyRefine(ConstArrayView<Int32> cells_local_id)
{
  IItemFamily* cell_family = m_mesh->cellFamily();
  Integer nb_cell = cells_local_id.size();
  info(4) << "Local_NbCellToRefine = " << nb_cell;
  Integer index = m_amr_patch_cell_groups.size();
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
    m_internal_api.cartesianMeshAMRPatchMng()->flagCellToRefine(cells_local_id);
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
