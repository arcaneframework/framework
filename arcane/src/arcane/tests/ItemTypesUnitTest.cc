// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypesUnitTest.cc                                        (C) 2000-2025 */
/*                                                                           */
/* Test des types des entités.                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/FactoryService.h"

#include "arcane/core/ItemTypeMng.h"
#include "arcane/core/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test du type des entités
 */
class ItemTypesUnitTest
: public BasicUnitTest
{
 public:

  explicit ItemTypesUnitTest(const ServiceBuildInfo& cb);

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  ItemTypeMng* m_item_type_mng = nullptr;

 private:

  void _doCheck(ItemTypeId item_type_id, Int16 type_id, const String& name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(ItemTypesUnitTest, IUnitTest, ItemTypesUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemTypesUnitTest::
ItemTypesUnitTest(const ServiceBuildInfo& mb)
: BasicUnitTest(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypesUnitTest::
_doCheck(ItemTypeId item_type_id, Int16 type_id, const String& expected_name)
{
  ItemTypeInfo* iti = m_item_type_mng->typeFromId(item_type_id);
  String type_name = iti->typeName();
  std::cout << "Item id=" << type_id << " name=" << type_name << "\n";
  if (item_type_id != type_id)
    ARCANE_FATAL("Bad ItemType type1={0} type2={1}", item_type_id, type_id);
  if (type_name != expected_name)
    ARCANE_FATAL("Bad ItemType name name={0} expected={1}", type_name, expected_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypesUnitTest::
initializeTest()
{
  m_item_type_mng = mesh()->itemTypeMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define TEST_ITEM_TYPE(a) _doCheck(ITI_##a, IT_##a, #a)
#define TEST_ITEM_TYPE_BAD(a) _doCheck(ITI_##a, IT_##a, (String("IT_") + #a))

void ItemTypesUnitTest::
executeTest()
{
  TEST_ITEM_TYPE(NullType);
  TEST_ITEM_TYPE(Vertex);
  TEST_ITEM_TYPE(Line2);
  TEST_ITEM_TYPE(Triangle3);
  TEST_ITEM_TYPE(Quad4);
  TEST_ITEM_TYPE(Pentagon5);
  TEST_ITEM_TYPE(Hexagon6);
  TEST_ITEM_TYPE(Tetraedron4);
  TEST_ITEM_TYPE(Pyramid5);
  TEST_ITEM_TYPE(Pentaedron6);
  TEST_ITEM_TYPE(Hexaedron8);
  TEST_ITEM_TYPE(Heptaedron10);
  TEST_ITEM_TYPE(Octaedron12);
  TEST_ITEM_TYPE(HemiHexa7);
  TEST_ITEM_TYPE(HemiHexa6);
  TEST_ITEM_TYPE(HemiHexa5);
  TEST_ITEM_TYPE(AntiWedgeLeft6);
  TEST_ITEM_TYPE(AntiWedgeRight6);
  TEST_ITEM_TYPE(DiTetra5);
  TEST_ITEM_TYPE(DualNode);
  TEST_ITEM_TYPE(DualEdge);
  TEST_ITEM_TYPE(DualFace);
  TEST_ITEM_TYPE(DualCell);
  TEST_ITEM_TYPE(Link);
  TEST_ITEM_TYPE(FaceVertex);
  TEST_ITEM_TYPE(CellLine2);
  TEST_ITEM_TYPE(DualParticle);
  TEST_ITEM_TYPE_BAD(Enneedron14);
  TEST_ITEM_TYPE_BAD(Decaedron16);
  TEST_ITEM_TYPE(Heptagon7);
  TEST_ITEM_TYPE(Octogon8);
  TEST_ITEM_TYPE(Line3);
  TEST_ITEM_TYPE(Triangle6);
  TEST_ITEM_TYPE(Quad8);
  TEST_ITEM_TYPE(Tetraedron10);
  TEST_ITEM_TYPE(Hexaedron20);
  TEST_ITEM_TYPE(Cell3D_Line2);
  TEST_ITEM_TYPE(Cell3D_Triangle3);
  TEST_ITEM_TYPE(Cell3D_Quad4);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
