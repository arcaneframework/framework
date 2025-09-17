// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NeoGraphTest.cpp                                (C) 2000-2025             */
/*                                                                           */
/* Test dag plug in Neo AlgorithmPropertyGraph                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <gtest/gtest.h>
#include "neo/Neo.h"
#include "neo/MeshKernel.h"

//----------------------------------------------------------------------------/

void _addAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph& mesh, Neo::Family& item_family, std::vector<int>& algo_order) {
  // Consume P1 produce P2
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ item_family, "prop1" },
                    Neo::MeshKernel::OutProperty{ item_family, "prop2" },
                    [&algo_order]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p1,
                                  [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p2) {
                      std::cout << "Algo 2 " << std::endl;
                      algo_order.push_back(2);
                    });
  // Produce P1
  mesh.addAlgorithm(Neo::MeshKernel::OutProperty{ item_family, "prop1" }, [&algo_order]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p1) {
    std::cout << "Algo 1 " << std::endl;
    algo_order.push_back(1);
  });
}

TEST(NeoGraphTest, BaseTest) {
  Neo::MeshKernel::AlgorithmPropertyGraph mesh{ "test_mesh" };
  Neo::Family cell_family{ Neo::ItemKind::IK_Cell, "cell_family" };
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop1");
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop2");
  std::vector<int> algo_index(3);
  std::vector<int> algo_order;

  _addAlgorithms(mesh, cell_family, algo_order);
  mesh.applyAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::FIFO);
  EXPECT_TRUE(std::equal(algo_order.begin(), algo_order.end(), std::vector{ 2, 1 }.begin()));
  algo_order.clear();

  _addAlgorithms(mesh, cell_family, algo_order);
  mesh.applyAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_TRUE(std::equal(algo_order.begin(), algo_order.end(), std::vector{ 1, 2 }.begin()));
  algo_order.clear();

  _addAlgorithms(mesh, cell_family, algo_order);
  mesh.applyAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::LIFO);
  EXPECT_TRUE(std::equal(algo_order.begin(), algo_order.end(), std::vector{ 1, 2 }.begin()));
  algo_order.clear();
}

//----------------------------------------------------------------------------/

TEST(NeoGraphTest, AlgoPersistanceTest) {
  Neo::MeshKernel::AlgorithmPropertyGraph mesh{ "test_mesh" };
  Neo::Family cell_family{ Neo::ItemKind::IK_Cell, "cell_family" };
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop1");
  int nb_called_algo1 = 0;
  int nb_called_algo2 = 0;
  mesh.addAlgorithm(
  Neo::MeshKernel::OutProperty{ cell_family, "prop1" }, [&nb_called_algo1]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p1) {
    std::cout << "Algo 1" << std::endl;
    nb_called_algo1 += 1;
  },
  Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmPersistence::KeepAfterExecution);
  mesh.addAlgorithm(
  Neo::MeshKernel::OutProperty{ cell_family, "prop1" }, [&nb_called_algo2]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p1) {
    std::cout << "Algo 2" << std::endl;
    nb_called_algo2 += 1;
  },
  Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmPersistence::DropAfterExecution);

  mesh.applyAlgorithms(); // Apply all algorithms
  EXPECT_EQ(nb_called_algo1, 1);
  EXPECT_EQ(nb_called_algo2, 1);

  mesh.applyAlgorithms(); // Apply only persistant algo : Algo1
  EXPECT_EQ(nb_called_algo1, 2);
  EXPECT_EQ(nb_called_algo2, 1);

  // Remove kept algorithm
  bool remove_kept_algorithm = true;
  mesh.removeAlgorithms(remove_kept_algorithm);
  mesh.applyAlgorithms(); // Does not call any algo
  EXPECT_EQ(nb_called_algo1, 2);
  EXPECT_EQ(nb_called_algo2, 1);

  // Add both algorithms again
  mesh.addAlgorithm(
  Neo::MeshKernel::OutProperty{ cell_family, "prop1" }, [&nb_called_algo1]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p1) {
    std::cout << "Algo 1" << std::endl;
    nb_called_algo1 += 1;
  },
  Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmPersistence::KeepAfterExecution);
  mesh.addAlgorithm(
  Neo::MeshKernel::OutProperty{ cell_family, "prop1" }, [&nb_called_algo2]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p1) {
    std::cout << "Algo 2" << std::endl;
    nb_called_algo2 += 1;
  },
  Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmPersistence::DropAfterExecution);
  mesh.applyAndKeepAlgorithms(); // Apply and keep both algos
  EXPECT_EQ(nb_called_algo1, 3);
  EXPECT_EQ(nb_called_algo2, 2);
  mesh.applyAndKeepAlgorithms(); // Apply and keep both algos
  EXPECT_EQ(nb_called_algo1, 4);
  EXPECT_EQ(nb_called_algo2, 3);
  mesh.removeAlgorithms(); // remove only non-persistent algo
  mesh.applyAlgorithms(); // Apply only Algo1
  EXPECT_EQ(nb_called_algo1, 5);
  EXPECT_EQ(nb_called_algo2, 3);
  mesh.removeAlgorithms(remove_kept_algorithm); // remove Algo1
  mesh.applyAlgorithms(); // not applying any algo
  EXPECT_EQ(nb_called_algo1, 5);
  EXPECT_EQ(nb_called_algo2, 3);
  mesh.applyAlgorithms(); // still not applying any algo (to check that persistent algorithms are not rescheduled by call to applyAlgorithms)
  EXPECT_EQ(nb_called_algo1, 5);
  EXPECT_EQ(nb_called_algo2, 3);
}

//----------------------------------------------------------------------------/

TEST(NeoGraphTest, OneProducingAlgoTest) {
  Neo::MeshKernel::AlgorithmPropertyGraph mesh{ "test_mesh" };
  Neo::Family cell_family{ Neo::ItemKind::IK_Cell, "cell_family" };
  bool is_called = false;

  // First try without adding property: algo not called
  mesh.addAlgorithm(Neo::MeshKernel::OutProperty{ cell_family, "prop1" }, [&is_called]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p1) {
    std::cout << "Algo 1" << std::endl;
    is_called = true;
  });
  mesh.applyAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_FALSE(is_called);

  // Now add property: algo must be called
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop1");
  mesh.addAlgorithm(Neo::MeshKernel::OutProperty{ cell_family, "prop1" }, [&is_called]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p1) {
    std::cout << "Algo 1" << std::endl;
    is_called = true;
  });

  mesh.applyAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_TRUE(is_called);
}

//----------------------------------------------------------------------------/

TEST(NeoGraphTest, OneConsumingProducingAlgoTest) {
  Neo::MeshKernel::AlgorithmPropertyGraph mesh{ "test_mesh" };
  Neo::Family cell_family{ Neo::ItemKind::IK_Cell, "cell_family" };
  bool is_called = false;
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop1", Neo::PropertyStatus::ExistingProperty }, // Existing means don't need to be computed by another algo
                    Neo::MeshKernel::OutProperty{ cell_family, "prop2" }, [&is_called]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p1, [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p2) {
                      std::cout << "Algo 1" << std::endl;
                      is_called = true;
                    });

  // First try without adding property: algo not called
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_FALSE(is_called);

  // Second try adding only produced property: algo still not called
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop2");
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_FALSE(is_called);

  // Third try  adding only produced property: algo still not called
  cell_family.removeProperties();
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop1");
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_FALSE(is_called);

  // Last try adding both properties: algo called
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop2");
  mesh.applyAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_TRUE(is_called);
  // Rk: if prop1 had been a ComputedProperty, the algo would not have been called, since no producing algo
  // see bellow
  is_called = false;
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop1", Neo::PropertyStatus::ComputedProperty }, // ComputedProperty is the default
                    Neo::MeshKernel::OutProperty{ cell_family, "prop2" }, [&is_called]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p1, [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p2) {
                      std::cout << "Algo 1" << std::endl;
                      is_called = true;
                    });
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_FALSE(is_called); // algo not called, since no one produces prop1, a ComputedProperty

  // if we add an algorithm producing prop1, it works
  mesh.addAlgorithm(Neo::MeshKernel::OutProperty{ cell_family, "prop1" },
                    []([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p1) {
                      std::cout << "Algo 0" << std::endl;
                    });
  mesh.applyAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_TRUE(is_called);

  // Detect cycle in graph : Produce and consume same property
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop1", Neo::PropertyStatus::ExistingProperty },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop1" }, []([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p1, [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p1_bis) {});
  EXPECT_THROW(mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG), std::runtime_error);
  // if algorithms are kept must still throw
  EXPECT_THROW(mesh.applyAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG), std::runtime_error);
  // Check that dag is cleared even when throws
  EXPECT_NO_THROW(mesh.applyAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG));
  // If the property Prop1 does not exist, the algo is not registered in the graph, no cycle is detected
  cell_family.removeProperties();
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop1", Neo::PropertyStatus::ExistingProperty },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop1" }, []([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p1, [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p1_bis) {});
  EXPECT_NO_THROW(mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG));
}

//----------------------------------------------------------------------------/

TEST(NeoGraphTest, OneAlgoMultipleAddTest) {
  Neo::MeshKernel::AlgorithmPropertyGraph mesh{ "test_mesh" };
  Neo::Family cell_family{ Neo::ItemKind::IK_Cell, "cell_family" };
  bool is_called = false;
  cell_family.addMeshScalarProperty<int>("prop1");
  cell_family.addMeshScalarProperty<int>("prop2");
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop1", Neo::PropertyStatus::ExistingProperty },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop2" }, [&is_called]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p1, [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p2) {
                      std::cout << "Algo 1" << std::endl;
                      is_called = true;
                    });
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_TRUE(is_called);
  is_called = false;
  // Algorithm is kept, the next algo creates a cycle
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop2", Neo::PropertyStatus::ExistingProperty },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop1" }, [&is_called]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p2, [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p1) {
                      std::cout << "Algo 1" << std::endl;
                      is_called = true;
                    });
  EXPECT_THROW(mesh.applyAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG), std::runtime_error);

  // Retry cleaning algo at each step
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop1", Neo::PropertyStatus::ExistingProperty },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop2" }, [&is_called]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p1, [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p2) {
                      std::cout << "Algo 1" << std::endl;
                      is_called = true;
                    });
  mesh.applyAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_TRUE(is_called);
  is_called = false;
  // Algorithm is not kept, the next algo does not create a cycle
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop2", Neo::PropertyStatus::ExistingProperty },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop1" }, [&is_called]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p2, [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p1) {
                      std::cout << "Algo 1" << std::endl;
                      is_called = true;
                    });
  EXPECT_NO_THROW(mesh.applyAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG));
  EXPECT_TRUE(is_called);
}

//----------------------------------------------------------------------------/

TEST(NeoGraphTest, TwoAlgorithmsTest) {
  Neo::MeshKernel::AlgorithmPropertyGraph mesh{ "test_mesh" };
  Neo::Family cell_family{ Neo::ItemKind::IK_Cell, "cell_family" };
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop1");
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop2");

  // Test Cycle
  // Consume P1 produce P2
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop1" },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop2" },
                    []([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p1,
                       [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p2) {
                      std::cout << "Algo 2 " << std::endl;
                    });
  // Produce P1 Consume P2
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop2" },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop1" },
                    []([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p1,
                       [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p2) {
                      std::cout << "Algo 1 " << std::endl;
                    });
  // Must throw since cycle in graph
  EXPECT_THROW(mesh.applyAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG), std::runtime_error);

  // Test 3 properties graph  -- Prop3(existing)--> Algo 1 --Prop4,Prop5(existing)--> Algo 2 --Prop6-->
  auto is_algo1_called = false;
  auto is_algo2_called = false;
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop3", Neo::PropertyStatus::ExistingProperty },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop4" },
                    [&is_algo1_called]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p3,
                                       [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p4) {
                      std::cout << "Algo 1 " << std::endl;
                      is_algo1_called = true;
                    });
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop5", Neo::PropertyStatus::ExistingProperty },
                    Neo::MeshKernel::InProperty{ cell_family, "prop4" },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop6" },
                    [&is_algo2_called]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p5,
                                       [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p4,
                                       [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p6) {
                      std::cout << "Algo 2 " << std::endl;
                      is_algo2_called = true;
                    });
  // No property => no algo called
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_FALSE(is_algo1_called);
  EXPECT_FALSE(is_algo2_called);
  // Add produced properties (P4,P6) still no algo called
  cell_family.addMeshScalarProperty<int>("prop4");
  cell_family.addMeshScalarProperty<int>("prop6");
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_FALSE(is_algo1_called);
  EXPECT_FALSE(is_algo2_called);
  // Add P3: trigger Algo 1
  cell_family.addMeshScalarProperty<int>("prop3");
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_TRUE(is_algo1_called);
  EXPECT_FALSE(is_algo2_called);
  is_algo1_called = false;
  // Add P5: trigger Algo2
  cell_family.addMeshScalarProperty<int>("prop5");
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_TRUE(is_algo1_called);
  EXPECT_TRUE(is_algo2_called);
}

//----------------------------------------------------------------------------/

TEST(NeoGraphTest,TwoAlgoMultipleAddTest) {
  Neo::MeshKernel::AlgorithmPropertyGraph mesh{ "test_mesh" };
  Neo::Family cell_family{ Neo::ItemKind::IK_Cell, "cell_family" };
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop1");
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop2");
  // Consume P1 produce P2
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop1" },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop2" },
                    []([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p1,
                       [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p2) {
                      std::cout << "Algo 2 " << std::endl;
                    });
  // Produce P1 Consume P2
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop2" },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop1" },
                    []([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p1,
                       [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p2) {
                      std::cout << "Algo 1 " << std::endl;
                    });
  // Must throw since cycle in graph
  EXPECT_THROW(mesh.applyAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG), std::runtime_error);
  // Algos are cleared, does not throw anymore
  EXPECT_NO_THROW(mesh.applyAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG));
  // Add new algo with no cycle. Check everything is well cleaned
  bool is_called = true;
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop2", Neo::PropertyStatus::ExistingProperty },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop1" },
                    [&is_called]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p1,
                                 [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p2) {
                      std::cout << "Algo 1 " << std::endl;
                      is_called = true;
                    });
  EXPECT_NO_THROW(mesh.applyAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG));
  EXPECT_TRUE(is_called);
}

//----------------------------------------------------------------------------/

TEST(NeoGraphTest, MultipleAlgoTest) {
  Neo::MeshKernel::AlgorithmPropertyGraph mesh{ "test_mesh" };
  Neo::Family cell_family{ Neo::ItemKind::IK_Cell, "cell_family" };
  bool is_called_1 = false;
  bool is_called_2 = false;
  bool is_called_3 = false;
  bool is_called_4 = false;
  bool is_called_5 = false;

  // Produce P1
  mesh.addAlgorithm(Neo::MeshKernel::OutProperty{ cell_family, "prop1" }, [&is_called_1]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p1) {
    std::cout << "Algo 1 " << std::endl;
    is_called_1 = true;
  });
  // Produce P2
  mesh.addAlgorithm(Neo::MeshKernel::OutProperty{ cell_family, "prop2" }, [&is_called_2]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p2) {
    std::cout << "Algo 2 " << std::endl;
    is_called_2 = true;
  });
  // Consume P1 & P2, produce P3
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop1" },
                    Neo::MeshKernel::InProperty{ cell_family, "prop2" },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop3" },
                    [&is_called_3]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p1,
                                   [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p2,
                                   [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p3) {
                      std::cout << "Algo 3 " << std::endl;
                      is_called_3 = true;
                    });
  // Consume P1, produce P4
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop1" },
                    Neo::MeshKernel::InProperty{ cell_family, "prop0", Neo::PropertyStatus::ExistingProperty },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop4" },
                    [&is_called_4]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p1,
                                   [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p0,
                                   [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p3) {
                      std::cout << "Algo 4 " << std::endl;
                      is_called_4 = true;
                    });
  // Consume P2, produce P5
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop2" },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop5" },
                    [&is_called_5]([[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32> const& p1,
                                   [[maybe_unused]] Neo::MeshScalarPropertyT<Neo::utils::Int32>& p3) {
                      std::cout << "Algo 5 " << std::endl;
                      is_called_5 = true;
                    });
  // No property existing: no algorithm called
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_FALSE(is_called_1 || is_called_2 || is_called_3 || is_called_4 || is_called_5);
  // Add property 1: algo 1 is called
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop1");
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_TRUE(is_called_1);
  is_called_1 = false;
  EXPECT_FALSE(is_called_2 || is_called_3 || is_called_4 || is_called_5);
  // Add property 2: algo 1 & 2 are called
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop2");
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_TRUE(is_called_1);
  is_called_1 = false;
  EXPECT_TRUE(is_called_2);
  is_called_2 = false;
  EXPECT_FALSE(is_called_3 || is_called_4 || is_called_5);
  // Add property 3: algo 1, 2 & 3 are called
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop3");
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_TRUE(is_called_1);
  is_called_1 = false;
  EXPECT_TRUE(is_called_2);
  is_called_2 = false;
  EXPECT_TRUE(is_called_3);
  is_called_3 = false;
  EXPECT_FALSE(is_called_4 || is_called_5);
  // Add property 4 and 0: algo 1, 2, 3 &4 are called
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop4");
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_FALSE(is_called_4); // missing prop0
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop0");
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_TRUE(is_called_1);
  is_called_1 = false;
  EXPECT_TRUE(is_called_2);
  is_called_2 = false;
  EXPECT_TRUE(is_called_3);
  is_called_3 = false;
  EXPECT_TRUE(is_called_4);
  is_called_4 = false;
  EXPECT_FALSE(is_called_4 || is_called_5);
  // Add property 5: algo 1, 2, 3, 4 & 5 are called
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop5");
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_TRUE(is_called_1);
  is_called_1 = false;
  EXPECT_TRUE(is_called_2);
  is_called_2 = false;
  EXPECT_TRUE(is_called_3);
  is_called_3 = false;
  EXPECT_TRUE(is_called_4);
  is_called_4 = false;
  EXPECT_TRUE(is_called_5);
  is_called_5 = false;
}

TEST(NeoGraphTest,CascadingRemoveAlgosTest)
{
  Neo::MeshKernel::AlgorithmPropertyGraph mesh{ "test_mesh" };
  Neo::Family cell_family{ Neo::ItemKind::IK_Cell, "cell_family" };
  // Check algos are well removed
  bool is_called_1 = false;
  bool is_called_2 = false;
  bool is_called_3 = false;
  // Consume P1, produce P2
  mesh.removeAlgorithms();
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop1");
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop2");
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop1" },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop2" },
                    [&is_called_1](Neo::MeshScalarPropertyT<Neo::utils::Int32> const&,
                                        Neo::MeshScalarPropertyT<Neo::utils::Int32>&) {
                      std::cout << "Algo 1 " << std::endl;
                      is_called_1 = true;
  });
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop3");
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop2" },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop3" },
                    [&is_called_2](Neo::MeshScalarPropertyT<Neo::utils::Int32> const&,
                                        Neo::MeshScalarPropertyT<Neo::utils::Int32>&) {
                      std::cout << "Algo 2 " << std::endl;
                      is_called_2 = true;
  });
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop4");
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop3" },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop4" },
                [&is_called_3](Neo::MeshScalarPropertyT<Neo::utils::Int32> const&,
                                    Neo::MeshScalarPropertyT<Neo::utils::Int32>&) {
                      std::cout << "Algo 3 " << std::endl;
                      is_called_3 = true;
  });
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  // P1 is not produced, nobody should be called
  EXPECT_FALSE(is_called_1 || is_called_2 || is_called_3);
}



//----------------------------------------------------------------------------/

TEST(NeoGraphTest, ItemAndConnectivityTest) {

  Neo::MeshKernel::AlgorithmPropertyGraph mesh{ "test_mesh" };
  // Create a cell and its nodes
  Neo::Family cell_family{ Neo::ItemKind::IK_Cell, "cell_family" };
  Neo::Family node_family{ Neo::ItemKind::IK_Node, "node_family" };
  // Add cell uid
  mesh.addAlgorithm(Neo::MeshKernel::OutProperty{ cell_family, cell_family.lidPropName() },
                    [](Neo::ItemLidsProperty& cell_lid_prop) {
                      std::cout << "-- Add cells --" << std::endl;
                      cell_lid_prop.append({ 42 });
                    });
  // Connect cell with its nodes
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, cell_family.lidPropName() },
                    Neo::MeshKernel::InProperty{ node_family, node_family.lidPropName() },
                    Neo::MeshKernel::OutProperty{ cell_family, "cell_to_nodes" },
                    [](Neo::ItemLidsProperty const& cell_lids,
                       Neo::ItemLidsProperty const& node_lids,
                       Neo::MeshArrayPropertyT<Neo::utils::Int32>& cell_to_nodes) {
                      std::cout << "-- Add cell to nodes connectivity -- " << std::endl;
                      // only one cell, connected to all node lids
                      cell_to_nodes.resize({ 8 });
                      cell_to_nodes.init(cell_lids.values(), node_lids.values().localIds());
                      cell_to_nodes.debugPrint();
                    });
  // Add nodes
  mesh.addAlgorithm(Neo::MeshKernel::OutProperty{ node_family, node_family.lidPropName() },
                    [](Neo::ItemLidsProperty& node_lids) {
                      std::cout << "-- Add nodes --" << std::endl;
                      node_lids.append({ 0, 1, 2, 3, 4, 5, 6, 7 });
                    });
  // Connect nodes with owning cell
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, cell_family.lidPropName() },
                    Neo::MeshKernel::InProperty{ node_family, node_family.lidPropName() },
                    Neo::MeshKernel::OutProperty{ node_family, "node_to_cell" },
                    [](Neo::ItemLidsProperty const& cell_lids,
                       Neo::ItemLidsProperty const& node_lids,
                       Neo::MeshScalarPropertyT<Neo::utils::Int32>& node_to_cell) {
                      std::cout << "-- Add node to cell connectivity -- " << std::endl;
                      // All nodes connected to the same cell
                      node_to_cell.init(node_lids.values(), cell_lids.values().localIds().back());
                      node_to_cell.debugPrint();
                    });
  // Try to call without creating properties for connectivities
  // Only node and cell creation occurs
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_EQ(cell_family.nbElements(), 1);
  EXPECT_EQ(node_family.nbElements(), 8);
  // Add cell_to_nodes property
  cell_family.addMeshArrayProperty<Neo::utils::Int32>("cell_to_nodes");
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  // Check items are not added twice
  EXPECT_EQ(cell_family.nbElements(), 1);
  EXPECT_EQ(node_family.nbElements(), 8);
  // Clear property cell_to_nodes to be able to call again its filling algorithm
  cell_family.getConcreteProperty<Neo::MeshArrayPropertyT<Neo::utils::Int32>>("cell_to_nodes").clear();
  // Add node_to_cell property
  node_family.addMeshScalarProperty<Neo::utils::Int32>("node_to_cell");
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  // Check items are note added twice
  EXPECT_EQ(cell_family.nbElements(), 1);
  EXPECT_EQ(node_family.nbElements(), 8);
  // Check ids
  std::vector node_ids{ 0, 1, 2, 3, 4, 5, 6, 7 };
  std::vector cell_ids{ 42 };
  auto created_node_ids = node_family._lidProp().values();
  EXPECT_TRUE(std::equal(created_node_ids.begin(), created_node_ids.end(), node_ids.begin()));
  EXPECT_EQ(cell_family._lidProp()._getLidFromUid(42), 0);
  // Check connectivities
  auto& cell_to_nodes = cell_family.getConcreteProperty<Neo::MeshArrayPropertyT<Neo::utils::Int32>>("cell_to_nodes");
  auto cell_to_nodes_view = cell_to_nodes.constView();
  EXPECT_TRUE(std::equal(cell_to_nodes_view.begin(), cell_to_nodes_view.end(), node_ids.begin()));
  auto& node_to_cell = node_family.getConcreteProperty<Neo::MeshScalarPropertyT<Neo::utils::Int32>>("node_to_cell");
  auto node_to_cell_view = node_to_cell.constView();
  std::vector cell_id{ 0, 0, 0, 0, 0, 0, 0, 0 };
  EXPECT_TRUE(std::equal(node_to_cell_view.begin(), node_to_cell_view.end(), cell_id.begin()));
}

//----------------------------------------------------------------------------/

TEST(NeoGraphTest,PropertyStatusTest) {
  Neo::MeshKernel::AlgorithmPropertyGraph mesh{ "test_mesh" };
  Neo::Family cell_family{ Neo::ItemKind::IK_Cell, "cell_family" };
  Neo::MeshKernel::PropertyHolder in_prop_holder{ cell_family, "in_prop1" };
  Neo::MeshKernel::PropertyHolder out_prop_holder{ cell_family, "out_prop1" };
  auto algo1 = [](Neo::MeshKernel::InProperty prop_in, Neo::MeshKernel::OutProperty prop_out) {};
  std::shared_ptr<Neo::MeshKernel::IAlgorithm> ialgo1 = std::make_shared<Neo::MeshKernel::AlgoHandler<decltype(algo1)>>(
    Neo::MeshKernel::InProperty{cell_family,"in_prop1", Neo::PropertyStatus::ExistingProperty},
    Neo::MeshKernel::OutProperty{cell_family,"out_prop1"}, (std::move(algo1)));
  auto algo2 = [](Neo::MeshKernel::InProperty prop_in, Neo::MeshKernel::OutProperty prop_out) {};
  std::shared_ptr<Neo::MeshKernel::IAlgorithm> ialgo2 = std::make_shared<Neo::MeshKernel::AlgoHandler<decltype(algo2)>>(
    Neo::MeshKernel::InProperty{cell_family,"in_prop1", Neo::PropertyStatus::ComputedProperty},
    Neo::MeshKernel::OutProperty{cell_family,"out_prop1"}, (std::move(algo2)));

  EXPECT_EQ(mesh.propertyStatus(in_prop_holder.uniqueName(), ialgo1),Neo::PropertyStatus::ExistingProperty);
  EXPECT_EQ(mesh.propertyStatus(in_prop_holder.uniqueName(), ialgo2),Neo::PropertyStatus::ComputedProperty);
  EXPECT_EQ(mesh.propertyStatus(out_prop_holder.uniqueName(), ialgo1),Neo::PropertyStatus::ComputedProperty);
}

//----------------------------------------------------------------------------/

TEST(NeoGraphTest,PropertyStatusRemovalBugTest) {
  // If algo removal in the graph doesn't take into account property status, Algo 2 won't be called
  Neo::MeshKernel::AlgorithmPropertyGraph mesh{ "test_mesh" };
  Neo::Family cell_family{ Neo::ItemKind::IK_Cell, "cell_family" };
  // Add property
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop1");
  cell_family.addMeshScalarProperty<Neo::utils::Int32>("prop2");
  // Add algo
  bool algo1_is_called = false;
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop1",Neo::PropertyStatus::ComputedProperty },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop2" },
                    [&algo1_is_called](Neo::MeshScalarPropertyT<Neo::utils::Int32> const&,
                       Neo::MeshScalarPropertyT<Neo::utils::Int32>& ) {
                      std::cout << "Algo 1 " << std::endl;
                      algo1_is_called = true;
                       }
  );
  bool algo2_is_called = false;
  mesh.addAlgorithm(Neo::MeshKernel::InProperty{ cell_family, "prop1",Neo::PropertyStatus::ExistingProperty },
                    Neo::MeshKernel::OutProperty{ cell_family, "prop2" },
                    [&algo2_is_called](Neo::MeshScalarPropertyT<Neo::utils::Int32> const&,
                       Neo::MeshScalarPropertyT<Neo::utils::Int32>& ) {
                      std::cout << "Algo 2 " << std::endl;
                      algo2_is_called = true;
                    }
  );
  mesh.applyAndKeepAlgorithms(Neo::MeshKernel::AlgorithmPropertyGraph::AlgorithmExecutionOrder::DAG);
  EXPECT_FALSE(algo1_is_called);
  EXPECT_TRUE(algo2_is_called);
}

//----------------------------------------------------------------------------/
//----------------------------------------------------------------------------/