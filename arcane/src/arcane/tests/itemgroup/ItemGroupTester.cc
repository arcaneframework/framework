// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupTester.cc                                   (C) 2000-2013 */
/*                                                                           */
/* Service du test des groupes d'items                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/BasicUnitTest.h"

#include "arcane/tests/itemgroup/ItemGroupTester_axl.h"

#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemGroupTester
: public ArcaneItemGroupTesterObject
{

public:

  ItemGroupTester(const ServiceBuildInfo& sb)
    : ArcaneItemGroupTesterObject(sb) {}
  
  ~ItemGroupTester() {}

 public:

  virtual void initializeTest();
  virtual void executeTest();
  
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_ITEMGROUPTESTER(ItemGroupTester,ItemGroupTester);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupTester::
initializeTest()
{ 
  info() << "init item group test";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupTester::
executeTest()
{
	info() << "execute item group test";

	info() << "Own group and Ghost group test";
	CellGroup own_cell_group = allCells().own();
	CellGroup ghost_cell_group = allCells().ghost();
	UniqueArray<Integer> cell_array;
	UniqueArray<Integer> allcells_array;

	ENUMERATE_CELL(icell,own_cell_group) {
		cell_array.add(icell->localId());
	}
	ENUMERATE_CELL(icell,ghost_cell_group) {
		cell_array.add(icell->localId());
	}
	ENUMERATE_CELL(icell,allCells()) {
		allcells_array.add(icell->localId());
	}

	Integer s1 = cell_array.size();
	Integer s2 = allcells_array.size();

	if (s1 != s2) {
		fatal() << "Own cell group has " << allCells().own().size() << " elements,"
						<< " Ghost cell group has " << allCells().ghost().size() << " elements,"
						<< " and Allcells group has " << s2 << " elements /= " << s1 << " = "
						<< allCells().own().size() << " + " << allCells().ghost().size();
	}
	info() << "Own group and Ghost group test successful";


	info() << "Interface group test";
	FaceGroup interface_group = allFaces().interface();

	ENUMERATE_FACE(iface,interface_group) {
		const Cell & bcell = iface->backCell();
		const Cell & fcell = iface->frontCell();
		const bool isBackOwn = bcell.isOwn();
		const bool isFrontOwn = fcell.isOwn();
		if ( isBackOwn == isFrontOwn )
			fatal() << "Face " << iface->localId() << " has two neighboring own cells or ghost cells "
					"instead of one of each type";
	}
	info() << "Interface group test successful";

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
