// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DoFNodeTestService.cc                                       (C) 2000-2022 */
/*                                                                           */
/* Service de test de création de DoF à partir de Node.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/BasicUnitTest.h"
#include "arcane/VariableTypes.h"
#include "arcane/ServiceFactory.h"
#include "arcane/IIndexedIncrementalItemConnectivityMng.h"
#include "arcane/IIndexedIncrementalItemConnectivity.h"
#include "arcane/IndexedItemConnectivityView.h"
#include "arcane/IMesh.h"

#include "arcane/mesh/DoFFamily.h"

#include "arcane/tests/ArcaneTestGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test des variables
 */
class DoFNodeTestService
: public BasicUnitTest
{
 public:

  explicit DoFNodeTestService(const ServiceBuildInfo& cb);

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  Ref<IIndexedIncrementalItemConnectivity> m_node_dof_connectivity;

 private:

  void
  _buildDoFs();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(DoFNodeTestService,
                        Arcane::ServiceProperty("DoFNodeTestService", Arcane::ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(Arcane::IUnitTest));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DoFNodeTestService::
DoFNodeTestService(const ServiceBuildInfo& sbi)
: BasicUnitTest(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DoFNodeTestService::
executeTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DoFNodeTestService::
initializeTest()
{
  _buildDoFs();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DoFNodeTestService::
_buildDoFs()
{
  IItemFamily* dof_family_interface = mesh()->findItemFamily(Arcane::IK_DoF, "DoFNode", true);
  mesh::DoFFamily* dof_family = ARCANE_CHECK_POINTER(dynamic_cast<mesh::DoFFamily*>(dof_family_interface));

  Int32 nb_dof_per_node = 3;

  // Create the DoFs
  Int64UniqueArray uids(ownNodes().size() * nb_dof_per_node);
  Int64 max_node_uid = mesh::DoFUids::getMaxItemUid(mesh()->nodeFamily());
  Int64 max_dof_uid = mesh::DoFUids::getMaxItemUid(dof_family);
  {
    Integer dof_index = 0;
    ENUMERATE_NODE (inode, ownNodes()) {
      for (Integer i = 0; i < nb_dof_per_node; ++i) {
        uids[dof_index] = mesh::DoFUids::uid(max_dof_uid, max_node_uid, inode->uniqueId().asInt64(), i);
        ++dof_index;
      }
    }
  }
  Int32UniqueArray dof_lids(uids.size());
  dof_family->addDoFs(uids, dof_lids);
  dof_family->endUpdate();
  info() << "NB_DOF=" << dof_family->allItems().size();

  // Création d'une connectivité Node->DoF
  m_node_dof_connectivity = mesh()->indexedConnectivityMng()->findOrCreateConnectivity(mesh()->nodeFamily(), dof_family, "DoFNode");
  auto* cn = m_node_dof_connectivity->connectivity();
  {
    Integer dof_index = 0;
    ENUMERATE_NODE (inode, ownNodes()) {
      NodeLocalId node = *inode;
      for (Integer i = 0; i < nb_dof_per_node; ++i) {
        cn->addConnectedItem(node, DoFLocalId(dof_lids[dof_index]));
        ++dof_index;
      }
    }
  }
  info() << "End build Dofs";

  // Remplit les 3 DoF par les coordonnées des noeuds
  info() << "Fill DoFs";
  VariableDoFReal dof_var(VariableBuildInfo(dof_family, "DofValues"));
  VariableNodeReal3& node_coords(mesh()->nodesCoordinates());
  IndexedNodeDoFConnectivityView node_dof(m_node_dof_connectivity->view());
  ENUMERATE_ (Node, inode, ownNodes()) {
    Node node = *inode;
    Real3 coord = node_coords[node];
    dof_var[node_dof.dofId(node,0)] = coord.x;
    dof_var[node_dof.dofId(node,1)] = coord.y;
    dof_var[node_dof.dofId(node,2)] = coord.z;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
