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
#include "arcane/IMeshStats.h"
#include "arcane/IItemFamily.h"
#include "arcane/IParticleFamily.h"
#include "arcane/anyitem/AnyItem.h"
#include "arcane/mesh/ItemFamily.h"

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

  void _createGraphOfDof();
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
  _createGraphOfDof();
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
void GraphUnitTest::
_createGraphOfDof()
{

  String particle_family_name = "ArcaneParticles";
  //  particles
  IItemFamily* particle_family = m_mesh->createItemFamily(IK_Particle, particle_family_name);

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

  Int64UniqueArray uids;

  IParallelMng* pm = subDomain()->parallelMng();
  Integer nb_own_cell = ownCells().size();
  Integer max_own_cell = pm->reduce(Parallel::ReduceMax, nb_own_cell);
  Integer nb_own_face = ownFaces().size();
  Integer max_own_face = pm->reduce(Parallel::ReduceMax, nb_own_face);
  Integer comm_rank = pm->commRank();

  Integer max_total = max_own_cell + max_own_face;

  Int64 first_uid = max_total * comm_rank;

  Int32UniqueArray cell_lids;

  ENUMERATE_CELL (icell, ownCells()) {
    const Cell& cell = *icell;
    cell_lids.add(cell.localId());
    uids.add(first_uid);
    ++first_uid;
  }

  Int32UniqueArray particles_lid(m_mesh->allCells().size());

  particle_family->toParticleFamily()->addParticles(uids, cell_lids, particles_lid);

  particle_family->endUpdate();

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

  // Création des links

  Integer nb_link = ownFaces().size();

  auto nb_dual_node_per_link = 2;

  Int64UniqueArray links_infos(nb_link * (nb_dual_node_per_link + 1));
  links_infos.fill(NULL_ITEM_UNIQUE_ID);

  Integer links_infos_index = 0;

  // On remplit le tableau links_infos
  ENUMERATE_FACE (iface, ownFaces()) {
    auto const& face = *iface;
    auto const& back_cell = iface->backCell();
    auto const& front_cell = iface->frontCell();

    // New test policy : add only internal face
    if (iface->isSubDomainBoundary()) {
      nb_link--;
      continue;
    }

    links_infos[links_infos_index++] = m_dualUid_mng.uniqueIdOf(face);

    if (!back_cell.null()) {
      links_infos[links_infos_index] = cell_to_dual_node_property[back_cell];
    }

    links_infos_index++;

    if (!front_cell.null()) {
      links_infos[links_infos_index] = cell_to_dual_node_property[front_cell];
    }

    links_infos_index++;
  }
  assert(links_infos_index == nb_link * (1 + nb_dual_node_per_link));
  info() << "Creation des link pour back_cell et front_cell links_infos"; //<<links_infos;
  //graphdofs->addLinks(nb_link,2,links_infos);
  links_infos.resize(nb_link * (1 + nb_dual_node_per_link));
  graph_modifier->addLinks(nb_link, nb_dual_node_per_link, links_infos);

  graphdofs->printLinks();

  // Création des links
  nb_link = ownCells().size();

  Integer nb_facemax = 6;

  //Int64UniqueArray links_infos2(nb_link*7);
  Int64UniqueArray links_infos2(nb_link * (nb_facemax + 1));
  links_infos2.fill(NULL_ITEM_UNIQUE_ID);

  links_infos_index = 0;

  // On remplit le tableau links_infos
  ENUMERATE_CELL (icell, ownCells()) {
    Cell cell = *icell;

    //  Take only cell with nb_facemax faces, number of dual node per link is constant...
    if (cell.faces().size() != nb_facemax) {
      nb_link--;
      continue;
    }

    links_infos2[links_infos_index++] = m_dualUid_mng.uniqueIdOf(cell);

    for ( Face face : cell.faces()) {
      links_infos2[links_infos_index] = face_to_dual_node_property[face];
      links_infos_index++;
    }
  }

  info() << "Creation des link pour cell et face_cell links_infos2"; //<<links_infos2;

  links_infos2.resize(nb_link * (nb_facemax + 1));
  graph_modifier->addLinks(nb_link, 6, links_infos2);

  graphdofs->printLinks();

  // Création des links
  nb_link = all_particles.size();

  Int64UniqueArray links_infos3(nb_link * 2);
  links_infos3.fill(NULL_ITEM_UNIQUE_ID);

  links_infos_index = 0;

  // On remplit le tableau links_infos
  ENUMERATE_PARTICLE (i_part, all_particles) {
    const Particle part = *i_part;
    const Cell& cell = i_part->cell();
    links_infos3[links_infos_index++] = m_dualUid_mng.uniqueIdOf(cell, part);
    links_infos3[links_infos_index++] = particule_to_dual_node_property[part];
  }

  info() << "Creation des link pour cell et particule links_infos3"; //<<links_infos3;

  graph_modifier->addLinks(nb_link, 1, links_infos3);

  graphdofs->printLinks();

  // Création des links
  nb_link = ownFaces().size();

  Int64UniqueArray links_infos4(nb_link * 3);
  links_infos4.fill(NULL_ITEM_UNIQUE_ID);

  links_infos_index = 0;

  // On remplit le tableau links_infos
  ENUMERATE_FACE (iface, ownFaces()) {
    auto const& face = *iface;
    auto const& back_cell = iface->backCell();
    auto const& front_cell = iface->frontCell();

    // New test policy : add only internal face
    if (iface->isSubDomainBoundary()) {
      nb_link--;
      continue;
    }

    if (!back_cell.null() && !front_cell.null()) {
      links_infos4[links_infos_index++] = math::max(m_dualUid_mng.uniqueIdOf(face, back_cell), m_dualUid_mng.uniqueIdOf(face, front_cell));
    }
    else if (!front_cell.null()) {
      links_infos4[links_infos_index++] = m_dualUid_mng.uniqueIdOf(face, front_cell);
    }
    else if (!back_cell.null()) {
      links_infos4[links_infos_index++] = m_dualUid_mng.uniqueIdOf(face, back_cell);
    }

    if (!back_cell.null()) {
      ENUMERATE_PARTICLE (i_part, all_particles) {
        const Particle part = *i_part;
        const Cell& cell = i_part->cell();
        if (cell == back_cell) {
          links_infos4[links_infos_index] = particule_to_dual_node_property[part];
        }
      }
    }

    links_infos_index++;

    if (!front_cell.null()) {
      ENUMERATE_PARTICLE (i_part, all_particles) {
        const Particle part = *i_part;
        const Cell& cell = i_part->cell();
        if (cell == front_cell) {
          links_infos4[links_infos_index] = particule_to_dual_node_property[part];
        }
      }
    }

    links_infos_index++;
  }
  info() << "Creation des link pour face cell et particule links_infos4"; //<<links_infos4;

  links_infos4.resize(nb_link * 3);
  graph_modifier->addLinks(nb_link, 2, links_infos4);

  graph_modifier->endUpdate();
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
