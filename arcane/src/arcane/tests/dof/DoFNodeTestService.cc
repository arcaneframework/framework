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

#include "arcane/mesh/DoFManager.h"

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

 public:

  //DoFManager& dofMng() { return m_dof_mng; }

 private:

  //DoFManager m_dof_mng;

 private:

  void _buildDoFs();
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

  // Done for node since it's easier to add new items
  //mesh::DoFFamily& dofs_on_node_family = dofMng().family(m_dofs_on_node_family_name);
  Int32 nb_dof_per_node = 3;
  // Create the DoFs
  Int64UniqueArray uids(ownNodes().size() * nb_dof_per_node);
  Int64 max_node_uid = mesh::DoFUids::getMaxItemUid(mesh()->nodeFamily());
  Int64 max_dof_uid = mesh::DoFUids::getMaxItemUid(dof_family);
  Integer j = 0;
  ENUMERATE_NODE (inode, ownNodes()) {
    for (Integer i = 0; i < nb_dof_per_node; ++i)
      uids[j++] = mesh::DoFUids::uid(max_dof_uid, max_node_uid, inode->uniqueId().asInt64(), i);
  }
  Int32UniqueArray lids(uids.size());
  dof_family->addDoFs(uids, lids);
  dof_family->endUpdate();
  info() << "NB_DOF=" << dof_family->allItems().size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
