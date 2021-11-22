// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NeoGraphTest                                    (C) 2000-2021             */
/*                                                                           */
/* Test dag plug in Neo MeshBase                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <gtest/gtest.h>
#include "neo/Neo.h"

TEST(NeoGraphTest,OneFamilyOnePropertyTest){
  Neo::MeshBase mesh{"test_mesh"};
  mesh.addFamily(Neo::ItemKind::IK_Cell, "cell_family");
  Neo::Family& cell_family = mesh.getFamily(Neo::ItemKind::IK_Cell, "cell_family");
  // Fill a property on the created cells, must be done after cell creation !
  cell_family.addProperty<Neo::utils::Int32>("prop");
  mesh.addAlgorithm(Neo::InProperty{cell_family,cell_family.lidPropName()},
                    Neo::OutProperty{cell_family,"prop"},
                    [](Neo::ItemLidsProperty const& cell_lid_prop, Neo::PropertyT<Neo::utils::Int32> prop){
                        std::cout << "Fill property after cell creation "<< std::endl;
                        prop.init(cell_lid_prop.values(),42);
                    });
  mesh.addAlgorithm(Neo::OutProperty{cell_family,cell_family.lidPropName()},
                    [](Neo::ItemLidsProperty & cell_lid_prop){
                        std::cout << "Create Cells "<< std::endl;
                        cell_lid_prop.append({0,1,2});
                    });
  mesh.applyAlgorithms();
  auto& prop = cell_family.getConcreteProperty<Neo::PropertyT<Neo::utils::Int32>>("prop");
  EXPECT_EQ(prop.size(),mesh.nbItems(Neo::ItemKind::IK_Cell));
  std::vector<int> ref_values(42, mesh.nbItems(Neo::ItemKind::IK_Cell));
  auto values = prop.values();
  EXPECT_TRUE(std::equal(values.begin(),values.end(),ref_values.begin()));
}