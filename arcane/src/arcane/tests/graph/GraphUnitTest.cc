// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GraphUnitTest.cc                                            (C) 2000-2006 */
/*                                                                           */
/* Service du test unitaire du graphe.                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include <cassert>
#include "arcane/BasicUnitTest.h"

#include "arcane/tests/graph/GraphUnitTest_axl.h"

#include "arcane/IItemOperationByBasicType.h"
#include "arcane/IParallelMng.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshModifier.h"
#include "arcane/IPrimaryMesh.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/IMeshStats.h"
#include "arcane/IItemFamily.h"
#include "arcane/IParticleFamily.h"
#include "arcane/anyitem/AnyItem.h"
#include "arcane/mesh/ItemFamily.h"
#include "arcane/mesh/ParticleFamily.h"

#include "arcane/mesh/DualUniqueIdMng.h"

#include "arcane/mesh/DoFManager.h"
#include "arcane/mesh/DoFFamily.h"
#include "arcane/mesh/ItemConnectivity.h"
#include "arcane/mesh/ItemConnectivityMng.h"
#include "arcane/mesh/GhostLayerFromConnectivityComputer.h"

#include "arcane/mesh/GraphDoFs.h"
#include "arcane/mesh/GraphBuilder.h"

#include "arcane/IItemConnectivitySynchronizer.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#include "arcane/MathUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test du graphe
 */
class GraphUnitTest
: public ArcaneGraphUnitTestObject
{
 public:

  class CountOperationByBasicType
  : public TraceAccessor
  , public IItemOperationByBasicType
  {
   public:

    explicit CountOperationByBasicType(ITraceMng* m)
    : TraceAccessor(m)
    {}

   public:

    void applyVertex(ItemVectorView group)
    {
      info() << "NB Vertex = " << group.size();
    }
    void applyLine2(ItemVectorView group)
    {
      info() << "NB Line2 = " << group.size();
    }
    void applyTriangle3(ItemVectorView group)
    {
      info() << "NB Triangle3 = " << group.size();
    }
    void applyQuad4(ItemVectorView group)
    {
      info() << "NB Quad4 = " << group.size();
    }
    void applyPentagon5(ItemVectorView group)
    {
      info() << "NB Pentagon5 = " << group.size();
    }
    void applyHexagon6(ItemVectorView group)
    {
      info() << "NB Hexagon6 = " << group.size();
    }
    void applyTetraedron4(ItemVectorView group)
    {
      info() << "NB Tetraedron4 = " << group.size();
    }
    void applyPyramid5(ItemVectorView group)
    {
      info() << "NB Pyramid5 = " << group.size();
    }
    void applyPentaedron6(ItemVectorView group)
    {
      info() << "NB Pentaedron6 = " << group.size();
    }
    void applyHexaedron8(ItemVectorView group)
    {
      info() << "NB Hexaedron8 = " << group.size();
    }
    void applyHeptaedron10(ItemVectorView group)
    {
      info() << "NB Heptaedron10 = " << group.size();
    }
    void applyOctaedron12(ItemVectorView group)
    {
      info() << "NB Octaedron12 = " << group.size();
    }
    void applyHemiHexa7(ItemVectorView group)
    {
      info() << "NB HemiHexa7 = " << group.size();
    }
    void applyHemiHexa6(ItemVectorView group)
    {
      info() << "NB HemiHexa6 = " << group.size();
    }
    void applyHemiHexa5(ItemVectorView group)
    {
      info() << "NB HemiHexa5 = " << group.size();
    }
    void applyAntiWedgeLeft6(ItemVectorView group)
    {
      info() << "NB AntiWedgeLeft6 = " << group.size();
    }
    void applyAntiWedgeRight6(ItemVectorView group)
    {
      info() << "NB AntiWedgeRight6 = " << group.size();
    }
    void applyDiTetra5(ItemVectorView group)
    {
      info() << "NB DiTetra5 = " << group.size();
    }
    void applyDualNode(ItemVectorView group)
    {
      info() << "NB DualNode = " << group.size();
    }
    void applyDualEdge(ItemVectorView group)
    {
      info() << "NB DualEdge = " << group.size();
    }
    void applyDualFace(ItemVectorView group)
    {
      info() << "NB DualFace = " << group.size();
    }
    void applyDualCell(ItemVectorView group)
    {
      info() << "NB DualCell = " << group.size();
    }
    void applyLink(ItemVectorView group)
    {
      info() << "NB Link = " << group.size();
    }
  };

  typedef ItemConnectivityT<Cell, DoF> CellToDoFConnectivity;
  typedef ItemArrayConnectivityT<Cell, DoF> CellToDoFsConnectivity;
  typedef ItemConnectivityT<Node, DoF> NodeToDoFConnectivity;
  typedef ItemArrayConnectivityT<Node, DoF> NodeToDoFsConnectivity;
  typedef ItemArrayConnectivityT<Face, DoF> FaceToDoFsConnectivity;
  typedef ItemMultiArrayConnectivityT<Face, DoF> FaceToDoFsMultiConnectivity;

 public:

  GraphUnitTest(const ServiceBuildInfo& mb)
  : ArcaneGraphUnitTestObject(mb)
  , m_mesh(mb.mesh())
  , m_stats(NULL)
  , m_connectivity_mng(mb.subDomain()->traceMng())
  , m_dof_mng(mb.mesh(), &m_connectivity_mng)
  , m_dualUid_mng(mb.subDomain()->traceMng())
  {}

  ~GraphUnitTest() { delete m_stats; }

 public:

  void initializeTest();
  void executeTest();

 private:

  IGraph2* _createGraphOfDof(IItemFamily* particle_family);
  void _checkGraphDofConnectivity(IGraph2* graph_dof);

 private:

  IMesh* m_mesh;

  IMeshStats* m_stats;

  ItemConnectivityMng m_connectivity_mng;
  DoFManager m_dof_mng;
  DoFManager& dofMng() { return m_dof_mng; }
  DualUniqueIdMng m_dualUid_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_GRAPHUNITTEST(GraphUnitTest, GraphUnitTest);

/*---------------------------------------------------------------------------*/
/*--------------------- ------------------------------------------------------*/

void GraphUnitTest::
executeTest()
{
  auto pm = subDomain()->parallelMng();
  Integer nb_own_cell = ownCells().size();
  Integer max_own_cell = pm->reduce(Parallel::ReduceMax, nb_own_cell);
  Integer nb_own_face = ownFaces().size();
  Integer max_own_face = pm->reduce(Parallel::ReduceMax, nb_own_face);
  Integer comm_rank = pm->commRank();

  Integer max_total = max_own_cell + max_own_face;

  Int64 first_uid = max_total * comm_rank;

  String particle_family_name = mesh::ParticleFamily::defaultFamilyName();
  //  particles
  IItemFamily* particle_family = m_mesh->createItemFamily(IK_Particle, particle_family_name);
  IParticleFamily* true_particle_family = dynamic_cast<IParticleFamily*>(particle_family);
  if (true_particle_family)
    true_particle_family->setEnableGhostItems(true);

  VariableItemInt32& cell_new_owners = mesh()->cellFamily()->itemsNewOwner();
  if (pm->commSize() > 1) {
    // SAVE ORIGINAL PARTITION MOVE ALL CELLS ON PROC 0
    ENUMERATE_CELL (icell, allCells()) {
      m_orig_cell_owner[icell] = icell->owner();
      cell_new_owners[icell] = 0; // everybody on subdomain 0
      info() << "Cell uid " << icell->uniqueId() << " has owner " << icell->owner();
    }
    mesh()->utilities()->changeOwnersFromCells();
    mesh()->modifier()->setDynamic(true);
    mesh()->toPrimaryMesh()->exchangeItems(); // update ghost is done.
  }

  Arcane::IGraph2* graphdofs = nullptr;
  if (pm->commRank() == 0) {
    info() << "CREATE PARTICLE ON MASTER";
    Int64UniqueArray uids;
    Int32UniqueArray cell_lids;

    ENUMERATE_CELL (icell, ownCells()) {
      const Cell& cell = *icell;
      cell_lids.add(cell.localId());
      uids.add(first_uid);
      ++first_uid;
    }

    Int32UniqueArray particles_lid(m_mesh->allCells().size());
    particle_family->toParticleFamily()->addParticles(uids, cell_lids, particles_lid);
  }

  particle_family->endUpdate();

  info() << "==================================================";
  info() << "CREATE GRAPH";
  graphdofs = _createGraphOfDof(particle_family);

  info() << "==================================================";
  info() << "Print DualNodes";
  graphdofs->printDualNodes();

  info() << "==================================================";
  info() << "Print Links";
  graphdofs->printLinks();

  info() << "==================================================";
  info() << "Check Connectivities";
  _checkGraphDofConnectivity(graphdofs);

  if (pm->commSize() > 1) {
    if (pm->commRank() == 0) {
      info() << "BROADCAST MESH FROM MASTER";
      // DISPACH CELL TO ORIGINAL PARTITION
      ENUMERATE_CELL (icell, allCells()) {
        cell_new_owners[icell] = m_orig_cell_owner[icell];
      }
      mesh()->utilities()->changeOwnersFromCells();
    }
    mesh()->modifier()->setDynamic(true);

    info() << "EXCHANGE ITEMS";
    mesh()->toPrimaryMesh()->exchangeItems(); // update ghost is done.

    info() << "MESH ENDUPDATE";
    mesh()->modifier()->endUpdate();

    info() << "GRAPH ENDUPDATE";
    graphdofs->modifier()->endUpdate();

    info() << "==================================================";
    info() << "Print DualNodes";
    graphdofs->printDualNodes();

    info() << "==================================================";
    info() << "Print Links";
    graphdofs->printLinks();

    info() << "==================================================";
    info() << "Check Connectivities";
    _checkGraphDofConnectivity(graphdofs);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphUnitTest::
initializeTest()
{
  m_stats = IMeshStats::create(traceMng(), mesh(), subDomain()->parallelMng());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
IGraph2* GraphUnitTest::
_createGraphOfDof(IItemFamily* particle_family)
{

  String particle_family_name = mesh::ParticleFamily::defaultFamilyName();

  auto graphdofs = mesh::GraphBuilder::createGraph(m_mesh, particle_family_name);

  IGraphModifier2* graph_modifier = graphdofs->modifier();

  // Cells

  Integer graph_nb_dual_node = m_mesh->nbCell();
  auto dual_node_kind = IT_DualCell;
  Int64UniqueArray cells_dual_nodes_infos(graph_nb_dual_node * 2);

  Integer infos_index = 0;
  ENUMERATE_CELL (icell, ownCells()) {
    const Cell& cell = *icell;

    Int64 dual_node_uid = m_dualUid_mng.uniqueIdOf(cell);

    cells_dual_nodes_infos[infos_index++] = dual_node_uid;

    Int64 dual_item_uid = icell->uniqueId().asInt64();

    cells_dual_nodes_infos[infos_index++] = dual_item_uid;
    info() << "CREATE DUALNODE[" << dual_node_uid << "] DUAL CELL : " << dual_item_uid;
  }

  //BD
  //graphdofs->addDualNodes(graph_nb_dual_node,dual_node_kind,cells_dual_nodes_infos);
  graph_modifier->addDualNodes(graph_nb_dual_node, dual_node_kind, cells_dual_nodes_infos);

  ItemScalarProperty<Int64> cell_to_dual_node_property;
  cell_to_dual_node_property.resize(m_mesh->cellFamily(), NULL_ITEM_LOCAL_ID);
  ENUMERATE_CELL (icell, ownCells()) {
    const Cell& cell = *icell;
    cell_to_dual_node_property[cell] = m_dualUid_mng.uniqueIdOf(cell);
  }

  // Faces
  graph_nb_dual_node = m_mesh->nbFace();
  dual_node_kind = IT_DualFace;
  Int64UniqueArray dual_nodes_infos(graph_nb_dual_node * 2);

  infos_index = 0;
  ENUMERATE_FACE (iface, ownFaces()) {
    const Face& face = *iface;
    Int64 dual_node_uid = m_dualUid_mng.uniqueIdOf(face);
    dual_nodes_infos[infos_index++] = dual_node_uid;
    Int64 dual_item_uid = iface->uniqueId().asInt64();
    dual_nodes_infos[infos_index++] = dual_item_uid;
    info() << "CREATE DUALNODE[" << dual_node_uid << "] DUAL FACE : " << dual_item_uid;
  }

  //graphdofs->addDualNodes(graph_nb_dual_node,dual_node_kind,dual_nodes_infos);
  graph_modifier->addDualNodes(graph_nb_dual_node, dual_node_kind, dual_nodes_infos);

  ItemScalarProperty<Int64> face_to_dual_node_property;
  face_to_dual_node_property.resize(m_mesh->faceFamily(), NULL_ITEM_LOCAL_ID);
  ENUMERATE_FACE (iface, ownFaces()) {
    const Face& face = *iface;
    Int64 dual_node_uid = m_dualUid_mng.uniqueIdOf(face);
    face_to_dual_node_property[face] = dual_node_uid;
  }

  graph_nb_dual_node = particle_family->allItems().size();
  dual_node_kind = IT_DualParticle;
  Int64UniqueArray particle_dual_nodes_infos(graph_nb_dual_node * 2);

  ParticleGroup all_particles(particle_family->allItems());

  infos_index = 0;
  ENUMERATE_PARTICLE (i_part, all_particles) {
    Particle part = *i_part;

    Int64 dual_node_uid = m_dualUid_mng.uniqueIdOf(part);

    particle_dual_nodes_infos[infos_index++] = dual_node_uid;

    Int64 dual_item_uid = part.uniqueId().asInt64();
    particle_dual_nodes_infos[infos_index++] = dual_item_uid;
    info() << "CREATE DUALNODE[" << dual_node_uid << "] DUAL PART : " << dual_item_uid;
  }

  //graphdofs->addDualNodes(graph_nb_dual_node,dual_node_kind,particle_dual_nodes_infos);
  graph_modifier->addDualNodes(graph_nb_dual_node, dual_node_kind, particle_dual_nodes_infos);

  ItemScalarProperty<Int64> particule_to_dual_node_property;
  particule_to_dual_node_property.resize(m_mesh->faceFamily(), NULL_ITEM_LOCAL_ID);
  ENUMERATE_PARTICLE (i_part, all_particles) {
    const Particle part = *i_part;
    Int64 dual_node_uid = m_dualUid_mng.uniqueIdOf(part);
    particule_to_dual_node_property[part] = dual_node_uid;
  }
  //
  graph_modifier->endUpdate();

  //////////////////////////////////////////////////////////////////////////////
  //
  // CREATION DE LINKS CELL-CELL
  //
  auto nb_dual_node_per_link = 2;
  {
    Integer nb_link = ownFaces().size();

    Int64UniqueArray links_infos(nb_link * (nb_dual_node_per_link + 1));
    links_infos.fill(NULL_ITEM_UNIQUE_ID);

    Integer links_infos_index = 0;
    Integer link_count = 0;

    // CREATION DE LINK CELL-CELL
    // On remplit le tableau links_infos
    ENUMERATE_FACE (iface, ownFaces()) {
      auto const& face = *iface;
      auto const& back_cell = iface->backCell();
      auto const& front_cell = iface->frontCell();

      // New test policy : add only internal face
      if (iface->isSubDomainBoundary()) {
        continue;
      }
      if (!back_cell.null() && !front_cell.null()) {
        links_infos[links_infos_index++] = m_dualUid_mng.uniqueIdOf<Cell, Cell>(back_cell, front_cell);
        links_infos[links_infos_index++] = cell_to_dual_node_property[back_cell];
        links_infos[links_infos_index++] = cell_to_dual_node_property[front_cell];
        info() << "CREATE LINK[" << m_dualUid_mng.uniqueIdOf<Cell, Cell>(back_cell, front_cell) << "] DUALCELL "
               << cell_to_dual_node_property[back_cell] << " DUALCELL " << cell_to_dual_node_property[front_cell];
        ++link_count;
      }
    }
    nb_link = link_count;
    links_infos.resize(nb_link * (1 + nb_dual_node_per_link));
    info() << "Creation des links CellCell pour links_infos"; //<<links_infos;
    graph_modifier->addLinks(nb_link, nb_dual_node_per_link, links_infos);
    graphdofs->printLinks();
  }

  //////////////////////////////////////////////////////////////////////////////
  //
  // CREATION DE LINK CELL-FACE
  //
  {
    Integer nb_facemax = 6;
    // Création des links
    Integer nb_link = ownCells().size() * nb_facemax;

    //Int64UniqueArray links_infos2(nb_link*7);
    Int64UniqueArray links_infos2(nb_link * (nb_dual_node_per_link + 1));
    links_infos2.fill(NULL_ITEM_UNIQUE_ID);

    Integer links_infos_index = 0;
    Integer link_count = 0;

    // On remplit le tableau links_infos
    ENUMERATE_CELL (icell, ownCells()) {
      Cell cell = *icell;

      //  Take only cell with nb_facemax faces, number of dual node per link is constant...
      if (cell.faces().size() != nb_facemax) {
        continue;
      }

      for (Face face : cell.faces()) {
        links_infos2[links_infos_index++] = m_dualUid_mng.uniqueIdOf(cell, face);
        links_infos2[links_infos_index++] = cell_to_dual_node_property[cell];
        links_infos2[links_infos_index++] = face_to_dual_node_property[face];
        info() << "CREATE LINK[" << m_dualUid_mng.uniqueIdOf(cell, face) << "] DUAL CELL "
               << cell_to_dual_node_property[cell] << " DUAL FACE " << face_to_dual_node_property[face];
        ++link_count;
      }
    }
    nb_link = link_count;
    info() << "Creation des links pour CellFace_cell links_infos2"; //<<links_infos2;

    links_infos2.resize(nb_link * (nb_dual_node_per_link + 1));
    graph_modifier->addLinks(nb_link, nb_dual_node_per_link, links_infos2);
    graphdofs->printLinks();
  }

  //////////////////////////////////////////////////////////////////////////////
  //
  // CREATION DE LINK CELL-PARTICLE
  //
  {
    Integer nb_link = all_particles.size();
    Int64UniqueArray links_infos3(nb_link * (nb_dual_node_per_link + 1));
    links_infos3.fill(NULL_ITEM_UNIQUE_ID);

    Integer links_infos_index = 0;
    Integer link_count = 0;
    // On remplit le tableau links_infos
    ENUMERATE_PARTICLE (i_part, all_particles) {
      const Particle part = *i_part;
      const Cell& cell = i_part->cell();
      links_infos3[links_infos_index++] = m_dualUid_mng.uniqueIdOf(cell, part);
      links_infos3[links_infos_index++] = cell_to_dual_node_property[cell];
      links_infos3[links_infos_index++] = particule_to_dual_node_property[part];
      ++link_count;
    }
    nb_link = link_count;
    links_infos3.resize(nb_link * (nb_dual_node_per_link + 1));
    info() << "Creation des link pour CellParticule links_infos3"; //<<links_infos3;
    graph_modifier->addLinks(nb_link, nb_dual_node_per_link, links_infos3);
    graphdofs->printLinks();
  }

  //////////////////////////////////////////////////////////////////////////////
  //
  // CREATION DE LINK FACE-PARTICLE
  //
  {
    Integer nb_link = ownFaces().size() * all_particles.size();

    Int64UniqueArray links_infos4(nb_link * (nb_dual_node_per_link + 1));
    links_infos4.fill(NULL_ITEM_UNIQUE_ID);

    Integer links_infos_index = 0;
    Integer link_count = 0;

    // On remplit le tableau links_infos
    ENUMERATE_FACE (iface, ownFaces()) {
      auto const& face = *iface;
      auto const& back_cell = iface->backCell();
      auto const& front_cell = iface->frontCell();

      if (iface->isSubDomainBoundary()) {
        continue;
      }

      if (!back_cell.null()) {
        ENUMERATE_PARTICLE (i_part, all_particles) {
          const Particle part = *i_part;
          const Cell& cell = i_part->cell();
          if (cell == back_cell) {
            links_infos4[links_infos_index++] = m_dualUid_mng.uniqueIdOf(face, part);
            links_infos4[links_infos_index++] = face_to_dual_node_property[face];
            links_infos4[links_infos_index++] = particule_to_dual_node_property[part];
            info() << "CREATE LINK[" << m_dualUid_mng.uniqueIdOf(face, part) << "] DUAL FACE "
                   << face_to_dual_node_property[face] << " DUAL PART " << particule_to_dual_node_property[part];
            ++link_count;
          }
        }
      }

      if (!front_cell.null()) {
        ENUMERATE_PARTICLE (i_part, all_particles) {
          const Particle part = *i_part;
          const Cell& cell = i_part->cell();
          if (cell == front_cell) {
            links_infos4[links_infos_index++] = m_dualUid_mng.uniqueIdOf(face, part);
            links_infos4[links_infos_index++] = face_to_dual_node_property[face];
            links_infos4[links_infos_index++] = particule_to_dual_node_property[part];
            ++link_count;
          }
        }
      }
    }
    nb_link = link_count;
    links_infos4.resize(nb_link * (nb_dual_node_per_link + 1));
    info() << "Creation des links pour FaceParticule links_infos4"; //<<links_infos4;
    graph_modifier->addLinks(nb_link, nb_dual_node_per_link, links_infos4);
    graphdofs->printLinks();
  }

  graph_modifier->endUpdate();

  return graphdofs;
}

void GraphUnitTest::
_checkGraphDofConnectivity(IGraph2* graph_dof)
{
  auto const& graph_connectivity = graph_dof->connectivity();
  ENUMERATE_DOF (idualnode, graph_dof->dualNodeFamily()->allItems()) {
    info() << "DualNode : lid = " << idualnode->localId();
    info() << "           uid = " << idualnode->uniqueId();
    auto links = graph_connectivity->links(*idualnode);
    for (auto const& link : links) {
      info() << "           Connected link : lid = " << link.localId();
      info() << "                            uid = " << link.uniqueId();
    }
  }
  // Pointwise access

  //ConnectivityItemVector dual_nodes(m_links_incremental_connectivity);
  ENUMERATE_DOF (i_link, graph_dof->linkFamily()->allItems()) {
    const DoF link = *i_link;
    info() << "Link = " << link.localId();
    info() << "          NB DUAL NODES : " << graph_connectivity->dualNodes(link).size();

    //ENUMERATE_DOF(idual_node,dual_nodes.connectedItems(link) )
    //auto dual_nodes = graph_connectivity->dualNodes(link) ;
    //for(Integer index=0;index<dual_nodes.size();++index)
    ENUMERATE_DOF (idual_node, graph_connectivity->dualNodes(link)) {
      //DoF const& dual_node = dual_nodes[index] ;
      DoF const& dual_node = *idual_node;
      info() << "     Dof : index = " << idual_node.index();
      info() << "     Dof : lid   = " << dual_node.localId();
      info() << "           uid   = " << dual_node.uniqueId();
      auto dual_item = graph_connectivity->dualItem(dual_node);
      info() << "           dual item : kind = " << dual_item.kind();
      info() << "                       lid  = " << dual_item.localId();
      info() << "                       uid  = " << dual_item.uniqueId();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//ARCANE_MESH_END_NAMESPACE
//ARCANE_END_NAMESPACE

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
