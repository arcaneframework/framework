// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMergeNodesUnitTest.cc                                   (C) 2000-2025 */
/*                                                                           */
/* Service de test de la fusion des noeuds.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ValueChecker.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/MeshStats.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/MeshMergeNodesUnitTest_axl.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de la fusion des noeuds.
 */
class MeshMergeNodesUnitTest
: public ArcaneMeshMergeNodesUnitTestObject
{
 public:

  class NodePair
  {
   public:

    NodePair()
    : left_uid(NULL_ITEM_UNIQUE_ID)
    , right_uid(NULL_ITEM_UNIQUE_ID)
    {}
    NodePair(Int64 left, Int64 right)
    : left_uid(left)
    , right_uid(right)
    {}
    Int64 left_uid;
    Int64 right_uid;
  };

 public:

  explicit MeshMergeNodesUnitTest(const ServiceBuildInfo& cb);
  ~MeshMergeNodesUnitTest();

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  using NodePairMap = std::map<Real2, NodePair>;
  NodePairMap m_node_pair_map;

 private:

  void _fillNodePairMap();
  void _addNodePairDirect(Real y, Int64 left_uid, Int64 right_uid);
  void _addNodePairDirect(Real2 yz, Int64 left_uid, Int64 right_uid);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MESHMERGENODESUNITTEST(MeshMergeNodesUnitTest,
                                               MeshMergeNodesUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMergeNodesUnitTest::
MeshMergeNodesUnitTest(const ServiceBuildInfo& mb)
: ArcaneMeshMergeNodesUnitTestObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMergeNodesUnitTest::
~MeshMergeNodesUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMergeNodesUnitTest::
executeTest()
{
  Int32UniqueArray nodes_local_id;
  Int32UniqueArray nodes_to_merge_local_id;

  IItemFamily* family = mesh()->nodeFamily();

  for (const auto& iter : m_node_pair_map) {
    ItemInternal* left_item = family->findOneItem(iter.second.left_uid);
    ItemInternal* right_item = family->findOneItem(iter.second.right_uid);
    if (!left_item && !right_item)
      continue;
    if (!left_item || !right_item)
      ARCANE_FATAL("Invalid NodePair (right or left item is null)");
    nodes_local_id.add(left_item->localId());
    nodes_to_merge_local_id.add(right_item->localId());
  }

  mesh()->utilities()->mergeNodes(nodes_local_id, nodes_to_merge_local_id);
  MeshStats ms(traceMng(), mesh(), mesh()->parallelMng());
  ms.dumpStats();

  ValueChecker vc(A_FUNCINFO);
  Int32 nb_node_pair = static_cast<Int32>(m_node_pair_map.size());
  vc.areEqual(nb_node_pair, options()->nbNodePair(), "NbNodePair");
  vc.areEqual(mesh()->allNodes().size(), options()->nbFinalNode(), "NbFinalNode");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMergeNodesUnitTest::
_fillNodePairMap()
{
  // Pour détecter les noeuds à connecter, on suppose que le maillage
  // a les propriétés suivantes:
  // - en 2D les noeuds à fusionner sont sur deux verticales, l'une correspond à
  // un X=0.5 et l'autre à X=0.51. Les noeuds de la seconde seront fusionnés
  // avec ceux de la première.
  // - en 3D, le principe est similaire mais la position de référence est 5.0
  // et 5.1.
  VariableNodeReal3& coords = mesh()->nodesCoordinates();
  const Real epsilon = 1.0e-4;
  Real ref_pos_left = 0.5;
  Real ref_pos_right = 0.51;
  if (mesh()->dimension() == 3) {
    ref_pos_left = 5.0;
    ref_pos_right = 5.1;
  }
  ENUMERATE_NODE (inode, allNodes()) {
    Node node = *inode;
    Real3 pos = coords[inode];
    Real2 pos_yz(pos.y, pos.z);
    if (math::isNearlyEqualWithEpsilon(pos.x, ref_pos_left, epsilon)) {
      info() << "Node Left =" << ItemPrinter(node) << " pos_yz=" << pos_yz;
      m_node_pair_map[pos_yz].left_uid = node.uniqueId();
    }
    if (math::isNearlyEqualWithEpsilon(pos.x, ref_pos_right, 1e-4)) {
      info() << "Node Right=" << ItemPrinter(node) << " pos_yz=" << pos_yz;
      m_node_pair_map[pos_yz].right_uid = node.uniqueId();
    }
  }
  info() << "NB_NODE_PAIR=" << m_node_pair_map.size();
  for (const auto& iter : m_node_pair_map) {
    Real2 pos_yz = iter.first;
    Int64 left_uid = iter.second.left_uid;
    Int64 right_uid = iter.second.right_uid;
    if (left_uid == NULL_ITEM_UNIQUE_ID)
      ARCANE_FATAL("No 'right' item for ypos={0} left={1}", pos_yz, left_uid);
    if (right_uid == NULL_ITEM_UNIQUE_ID)
      ARCANE_FATAL("No 'left' item for ypos={0} right={1}", pos_yz, right_uid);
    info() << "UID yz=" << pos_yz << " Left=" << left_uid
           << " Right=" << right_uid;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMergeNodesUnitTest::
_addNodePairDirect(Real y, Int64 left_uid, Int64 right_uid)
{
  m_node_pair_map.insert(std::make_pair(Real2(y, 0.0), NodePair(left_uid, right_uid)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMergeNodesUnitTest::
_addNodePairDirect(Real2 yz, Int64 left_uid, Int64 right_uid)
{
  m_node_pair_map.insert(std::make_pair(yz, NodePair(left_uid, right_uid)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMergeNodesUnitTest::
initializeTest()
{
  if (mesh()->parallelMng()->isParallel()) {
    if (mesh()->dimension()!=2)
      ARCANE_FATAL("This test is only valid for specific 'merge_nodes_2d.vtk'");
    // En parallèle, comme on ne peut pas déterminer la liste des noeuds
    // connectés car ils ne sont pas forcément dans le même sous-domaine
    // on remplit directement les valeurs. Du coup ce test ne fonctionne
    // qu'avec un maillage spécifique qui est actuellement le maillage 2D
    // NOTE: pour l'insant le test n'est pas actif en parallèle car il faut ajouter
    // des contraintes pour être sur que deux faces fusionnés sont dans le
    // même sous-domaine.
    _addNodePairDirect(-0.5, 6, 3);
    _addNodePairDirect(-0.4, 62, 35);
    _addNodePairDirect(-0.3, 63, 36);
    _addNodePairDirect(-0.2, 64, 37);
    _addNodePairDirect(-0.1, 65, 38);
    _addNodePairDirect(0.0, 66, 39);
    _addNodePairDirect(0.1, 67, 40);
    _addNodePairDirect(0.2, 68, 41);
    _addNodePairDirect(0.3, 69, 42);
    _addNodePairDirect(0.4, 70, 43);
    _addNodePairDirect(0.5, 7, 0);
  }
  else
    _fillNodePairMap();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
