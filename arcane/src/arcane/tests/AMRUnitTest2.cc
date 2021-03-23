// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRUnitTest.cc                                              (C) 2000-2010 */
/*                                                                           */
/* Service de test du raffinement/dérafinnement facon AMR.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/utils/StringBuilder.h"

#include "arcane/BasicUnitTest.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/AMRUnitTest2_axl.h"

#include "arcane/IMesh.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/IItemFamily.h"
#include "arcane/IMeshModifier.h"
#include "arcane/IMeshWriter.h"
#include "arcane/ServiceFinder.h"

#include "arcane/IPostProcessorWriter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test du maillage
 */
class AMRUnitTest2
: public ArcaneAMRUnitTest2Object
{
public:

public:

  AMRUnitTest2(const ServiceBuildInfo& cb);
  ~AMRUnitTest2();

 public:

  virtual void initializeTest();
  virtual void executeTest();

 private:

  void _writeMesh(const String& filename);
  void _refine(Integer nb_to_refine);
  void _postProcessAMR();

 private:
	 IMesh * new_mesh;
	 // Post-processing
	 RealUniqueArray times;
	 VariableCellReal * new_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_AMRUNITTEST2(AMRUnitTest2,AMRUnitTest2);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRUnitTest2::
AMRUnitTest2(const ServiceBuildInfo& mb)
: ArcaneAMRUnitTest2Object(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRUnitTest2::
~AMRUnitTest2()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRUnitTest2::
executeTest()
{
  _refine(10);
  new_mesh = subDomain()->mesh();
  new_data = new VariableCellReal(Arcane::VariableBuildInfo(new_mesh, "Data", new_mesh->cellFamily()->name(), Arcane::IVariable::PNoDump|Arcane::IVariable::PNoNeedSync));
  StringBuilder filename = "amrmesh";
  filename += subDomain()->subDomainId();
  filename += ".mli";
  _writeMesh(filename.toString());
  _postProcessAMR();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRUnitTest2::
_refine(Integer nb_to_refine)
{
//! AMR
	// Recherche les nb_to_refine premières mailles de type IT_Hexaedron8
	ENUMERATE_CELL(icell,ownCells()){
		Cell cell = *icell;
		ItemInternal* iitem = cell.internal();
		if (cell.type()==IT_Hexaedron8){
			Integer f = iitem->flags();
			f |= ItemInternal::II_Refine;
			iitem->setFlags(f);
			nb_to_refine--;
			if (nb_to_refine == 0)
				break;
		}
	}

	mesh()->modifier()->refineItems();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRUnitTest2::
initializeTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRUnitTest2::
_writeMesh(const String& filename)
{
  IMeshWriter* mesh_io =
  ServiceFinderT<IMeshWriter>::find(subDomain()->serviceMng(),"Lima");
  if (mesh_io)
    mesh_io->writeMeshToFile(mesh(),filename);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRUnitTest2::
_postProcessAMR()
{
  info() << "Post-process AMR \n ";
  IPostProcessorWriter* post_processor = options()->format();
  times.add(m_global_time());
  post_processor->setTimes(times);
  post_processor->setMesh(new_mesh);

  new_data->fill(0.);
  ENUMERATE_CELL(icell,allCells()) {
    (*new_data)[icell] = icell->owner();
  }

  VariableList variables;
  variables.add(new_data->variable());
  post_processor->setVariables(variables);
  ItemGroupList groups;
  groups.add(allCells());
  post_processor->setGroups(groups);
  IVariableMng * vm = subDomain()->variableMng();
  vm->writePostProcessing(post_processor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
