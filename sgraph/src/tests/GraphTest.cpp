// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GrapheTest                                     (C) 2000-2022              */
/*                                                                           */
/* Graph tests                                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <iostream>

#include <string>

#include <gtest/gtest.h>

#include "sgraph/DirectedGraph.h"
#include "sgraph/DirectedAcyclicGraph.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct Property
{
  std::string name;
  bool operator<(const Property& prop) const {
    return name < prop.name;
  }
};

struct Algorithm
{
  std::string name;
  bool operator<(const Algorithm& prop) const {
    return name < prop.name;
  }
};

std::ostream& operator<<(std::ostream& os, Algorithm* algo) {
  os << algo->name;
  return os;
}

std::ostream& operator<<(std::ostream& os, Property* prop) {
  os << prop->name;
  return os;
}

std::ostream& operator<<(std::ostream& os, Algorithm algo) {
  os << algo.name;
  return os;
}

std::ostream& operator<<(std::ostream& os, Property prop) {
  os << prop.name;
  return os;
}

TEST(DirectedGraphTest, UnitTest) {

  SGraph::DirectedGraph<Property, Algorithm> directed_graph{};
  auto prop1 = Property{ "prop_in" };
  auto prop2 = Property{ "prop_out" };
  auto prop3 = Property{ "prop_final" };
  auto algo1 = Algorithm{ "algo1" };
  auto algo2 = Algorithm{ "algo2" };
  directed_graph.addEdge(prop2, prop3, algo2);
  directed_graph.addEdge(prop1, prop2, algo1);
  directed_graph.print();
  auto edge = directed_graph.getEdge(Property{ "prop_in" }, Property{ "prop_out" });
  auto edge2 = directed_graph.getEdge(Property{ "prop_out" }, Property{ "prop_final" });
  EXPECT_EQ(edge->name, algo1.name);
  EXPECT_EQ(edge2->name, algo2.name);
  auto null_edge = directed_graph.getEdge(Property{ "a" }, Property{ "b" });
  EXPECT_EQ(null_edge, nullptr);
  directed_graph.clear();
  EXPECT_EQ(directed_graph.edges().size(),0);
  EXPECT_EQ(directed_graph.vertices().size(),0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(DirectedGraphTest, VertexTest) {
  SGraph::DirectedGraph<int, int> int_directed_graph{};
  int_directed_graph.addVertex(1);
  int_directed_graph.addVertex(2);
  int_directed_graph.addVertex(2);
  int_directed_graph.addVertex(3);
  int_directed_graph.addVertex(4);
  int_directed_graph.addVertex(5);
  int_directed_graph.addVertex(5);
  // check vertices() method
  std::vector<int> graph_vertices;
  for (auto vertex : int_directed_graph.vertices()) {
    graph_vertices.push_back(vertex);
  }
  std::vector graph_vertices_ref{ 1, 2, 3, 4, 5 };
  EXPECT_EQ(graph_vertices.size(), graph_vertices_ref.size());
  EXPECT_TRUE(std::equal(graph_vertices.begin(), graph_vertices.end(), graph_vertices_ref.begin()));
  // check const version
  auto const& const_graph = int_directed_graph;
  graph_vertices.clear();
  for (auto vertex : const_graph.vertices()) {
    graph_vertices.push_back(vertex);
  }
  EXPECT_EQ(graph_vertices.size(), graph_vertices_ref.size());
  EXPECT_TRUE(std::equal(graph_vertices.begin(), graph_vertices.end(), graph_vertices_ref.begin()));
  int_directed_graph.clear();
  EXPECT_EQ(int_directed_graph.vertices().size(),0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(DirectedGraphTest, EdgeTest) {

  SGraph::DirectedGraph<std::string, std::string> directed_graph{};
  directed_graph.addEdge("a", "b", "ab");
  directed_graph.addEdge("e", "g", "eg");

  std::cout << "getEdge (\"e\",\"g\") " << *directed_graph.getEdge("e", "g") << std::endl;
  std::cout << "getEdge (\"a\",\"b\") " << *directed_graph.getEdge("a", "b") << std::endl;

  std::cout << "sourceVertex(\"eg\") " << *directed_graph.getSourceVertex("eg") << std::endl;
  std::cout << "targetVertex(\"eg\") " << *directed_graph.getTargetVertex("eg") << std::endl;

  EXPECT_EQ(*directed_graph.getEdge("e", "g"), "eg");
  EXPECT_EQ(*directed_graph.getEdge("a", "b"), "ab");
  EXPECT_EQ(*directed_graph.getSourceVertex("eg"), "e");
  EXPECT_EQ(*directed_graph.getTargetVertex("eg"), "g");

  directed_graph.print();

  //directed_graph.addEdge("a", "b", "ab");
  directed_graph.addEdge("a", "d", "ad");
  directed_graph.addEdge("b", "e", "be");
  directed_graph.addEdge("c", "e", "ce");
  directed_graph.addEdge("a", "c", "ac");
  directed_graph.addEdge("e", "f", "ef");
  directed_graph.addEdge("g", "h", "gh");
  directed_graph.addEdge("f", "h", "fh");

  EXPECT_THROW(directed_graph.addEdge("f", "h", "fh"), std::runtime_error);

  std::cout << "Edge (a,b) " << *directed_graph.getEdge("a", "b") << std::endl;
  std::cout << "Edge (a,b) " << *directed_graph.getEdge("e", "g") << std::endl;
  std::cout << "Edge (a,d) " << *directed_graph.getEdge("a", "d") << std::endl;
  std::cout << "Edge (b,e) " << *directed_graph.getEdge("b", "e") << std::endl;
  std::cout << "Edge (c,e) " << *directed_graph.getEdge("c", "e") << std::endl;
  std::cout << "Edge (a,c) " << *directed_graph.getEdge("a", "c") << std::endl;
  std::cout << "Edge (e,f) " << *directed_graph.getEdge("e", "f") << std::endl;
  std::cout << "Edge (g,h) " << *directed_graph.getEdge("g", "h") << std::endl;
  std::cout << "Edge (f,h) " << *directed_graph.getEdge("f", "h") << std::endl;
  //
  std::cout << "Edge eg contains nodes " << *directed_graph.getSourceVertex("eg") << " " << *directed_graph.getTargetVertex("eg") << std::endl;
  std::cout << "Edge ab contains nodes " << *directed_graph.getSourceVertex("ab") << " " << *directed_graph.getTargetVertex("ab") << std::endl;

  // Check edge groups
  std::vector<std::string> edges;
  for (auto edge : directed_graph.edges()) {
    edges.push_back(edge);
  }
  auto ref_edges = { "ab", "eg", "ad", "be", "ce", "ac", "ef", "gh", "fh" };
  EXPECT_TRUE(std::equal(edges.begin(), edges.end(), ref_edges.begin()));

  edges.clear();
  for (auto edge : directed_graph.inEdges("h")) {
    edges.push_back(edge);
  }
  ref_edges = { "gh", "fh" };
  EXPECT_TRUE(std::equal(edges.begin(), edges.end(), ref_edges.begin()));

  edges.clear();
  for (auto edge : directed_graph.outEdges("a")) {
    edges.push_back(edge);
  }
  ref_edges = { "ab", "ad", "ac" };
  EXPECT_TRUE(std::equal(edges.begin(), edges.end(), ref_edges.begin()));

  directed_graph.clear();
  EXPECT_EQ(directed_graph.edges().size(),0);
  EXPECT_EQ(directed_graph.inEdges("h").size(),0);
  EXPECT_EQ(directed_graph.outEdges("a").size(),0);
}

/*---------------------------------------------------------------------------*/

TEST(DirectedGraphTest, EdgeTestConst) {
  SGraph::DirectedGraph<std::string, std::string> directed_graph{};
  directed_graph.addEdge("a", "b", "ab");
  directed_graph.addEdge("e", "g", "eg");
  auto const& const_graph = directed_graph;

  EXPECT_EQ(*const_graph.getEdge("e", "g"), "eg");
  EXPECT_EQ(*const_graph.getEdge("a", "b"), "ab");
  EXPECT_EQ(*const_graph.getSourceVertex("eg"), "e");
  EXPECT_EQ(*const_graph.getTargetVertex("eg"), "g");

  std::vector<std::string> edges;
  for (auto edge : const_graph.edges()) {
    edges.push_back(edge);
  }
  auto ref_edges = { "ab", "eg" };
  EXPECT_TRUE(std::equal(edges.begin(), edges.end(), ref_edges.begin()));

  edges.clear();
  for (auto edge : const_graph.inEdges("b")) {
    edges.push_back(edge);
  }
  ref_edges = { "ab" };
  EXPECT_TRUE(std::equal(edges.begin(), edges.end(), ref_edges.begin()));

  edges.clear();
  for (auto edge : const_graph.outEdges("b")) {
    edges.push_back(edge);
  }
  ref_edges = { "ab" };
  EXPECT_TRUE(std::equal(edges.begin(), edges.end(), ref_edges.begin()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, typename U, typename VertexEqualityPredicate, typename EdgeEqualityPredicate>
void _checkDag(std::vector<U> const& sorted_vertices_ref,
               std::vector<T> const& spanning_tree_edges_ref,
               SGraph::DirectedAcyclicGraph<U, T>& dag,
               VertexEqualityPredicate&& vertex_equality,
               EdgeEqualityPredicate&& edge_equality) {
  dag.print();

  using SGraphType = SGraph::DirectedAcyclicGraph<U, T>;

  // Print topologically oredered graph
  std::vector<typename SGraphType::vertex_type> sorted_vertices;
  for (const typename SGraphType::vertex_type& sorted_vertex : dag.topologicalSort())
  // Warning, topological sort returns std::reference_wrapper. Use correct type and not auto
  {
    std::cout << "Sorted Graph has vertex " << sorted_vertex << std::endl;
    sorted_vertices.push_back(sorted_vertex);
  }
  EXPECT_TRUE(std::equal(sorted_vertices.begin(), sorted_vertices.end(), sorted_vertices_ref.begin(), vertex_equality));

  // Print topologically oredered graph in reverse order
  sorted_vertices.clear();
  for (const typename SGraphType::vertex_type& sorted_vertex : dag.topologicalSort().reverseOrder()) {
    std::cout << "Reverse order sorted Graph has vertex " << sorted_vertex << std::endl;
    sorted_vertices.push_back(sorted_vertex);
  }
  EXPECT_TRUE(std::equal(sorted_vertices.begin(), sorted_vertices.end(), sorted_vertices_ref.rbegin(), vertex_equality));

  // Print Spanning tree (arbre couvrant)
  std::vector<typename SGraphType::edge_type> spanning_tree_edges;
  for (const typename SGraphType::edge_type& edge_tree : dag.spanningTree()) {
    std::cout << "Spanning tree has edge " << edge_tree << std::endl;
    spanning_tree_edges.push_back(edge_tree);
  }
  EXPECT_TRUE(std::equal(spanning_tree_edges.begin(), spanning_tree_edges.end(), spanning_tree_edges_ref.begin(), edge_equality));

  // Print Spanning tree (arbre couvrant) in reverse order
  spanning_tree_edges.clear();
  for (const typename SGraphType::edge_type& edge_tree : dag.spanningTree().reverseOrder()) {
    std::cout << "Reverse order spanning tree has edge " << edge_tree << std::endl;
    spanning_tree_edges.push_back(edge_tree);
  }
  EXPECT_TRUE(std::equal(spanning_tree_edges.begin(), spanning_tree_edges.end(), spanning_tree_edges_ref.rbegin(), edge_equality));
}

/*---------------------------------------------------------------------------*/

TEST(DirectedAcyclicGraphTest, UnitTest) {
  // Same code base as DirectedGraphT, the topological sort is added
  using SGraphType = SGraph::DirectedAcyclicGraph<std::string, std::string>;
  SGraphType dag{};
  dag.addVertex("aa");
  dag.addEdge("a", "h", "ah");
  dag.addEdge("e", "g", "eg");
  dag.addEdge("a", "b", "ab");
  dag.addEdge("a", "d", "ad");
  dag.addEdge("b", "e", "be");
  dag.addEdge("c", "e", "ce");
  dag.addEdge("a", "c", "ac");
  dag.addEdge("e", "f", "ef");
  dag.addEdge("g", "h", "gh");
  dag.addEdge("f", "h", "fh");
  dag.addVertex("aaa");

  std::vector<SGraphType::vertex_type> sorted_vertices_ref{ "aa", "a", "aaa", "b", "d", "c", "e", "g", "f", "h" };
  std::vector<SGraphType::edge_type> spanning_tree_edges_ref{ "ab", "ad", "ac", "be", "ce", "eg", "ef", "gh", "fh" };

  auto string_equality = [](std::string const& a, std::string const& b) { return a == b; };
  _checkDag(sorted_vertices_ref, spanning_tree_edges_ref, dag, string_equality, string_equality);

  // add edge and check impact
  dag.addEdge("a", "dprime", "adprime");
  dag.addEdge("dprime", "g", "dprimeg");
  dag.addEdge("h", "i", "hi");

  for (const std::string& sorted_vertex : dag.topologicalSort()) {
    std::cout << "Sorted Graph has vertex " << sorted_vertex << std::endl;
  }

  // Print Spanning tree (arbre couvrant)
  for (const std::string& edge_tree : dag.spanningTree()) {
    std::cout << "Spanning tree has edge " << edge_tree << std::endl;
  }

  dag.print();

  // Corrupt the dag inserting a cycle (topologicalSort() and print() will throw runtime_error)
  dag.addEdge("g", "a", "ga");
  dag.addEdge("b", "g", "bg");

  EXPECT_TRUE(dag.hasCycle());
  EXPECT_THROW(dag.topologicalSort(),std::runtime_error);

  // The graph is now corrupted...

  dag.clear();
  EXPECT_NO_THROW(dag.topologicalSort());
  EXPECT_EQ(dag.topologicalSort().size(),0);
  EXPECT_EQ(dag.spanningTree().size(),0);
  // Check clear is full. Add a regular edge. sort must not throw
  dag.addEdge("b","a","ba");
  EXPECT_NO_THROW(dag.topologicalSort());
  EXPECT_FALSE(dag.hasCycle());

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(DirectedAcyclicGraphTest, UserClassTest) {

  SGraph::DirectedAcyclicGraph<Algorithm, Property> algo_dag;
  algo_dag.addVertex(Algorithm{ "algo0" });
  algo_dag.addEdge(Algorithm{ "algo3" }, Algorithm{ "algo4" }, Property{ "P3" });
  algo_dag.addEdge(Algorithm{ "algo1" }, Algorithm{ "algo2" }, Property{ "P1" });
  algo_dag.addEdge(Algorithm{ "algo2" }, Algorithm{ "algo4" }, Property{ "P2" });
  algo_dag.addEdge(Algorithm{ "algo1" }, Algorithm{ "algo3" }, Property{ "P1" });
  algo_dag.addVertex(Algorithm{ "algo1-0" });

  for (Algorithm const& algo : algo_dag.topologicalSort()) {
    std::cout << "-- " << algo.name << std::endl;
  }
  algo_dag.print();
  auto sorted_vertices = algo_dag.topologicalSort();
  auto algo_equality = [](Algorithm const& a, Algorithm const& b) { return a.name == b.name; };
  auto prop_equality = [](Property const& a, Property const& b) { return a.name == b.name; };
  _checkDag(std::vector{ Algorithm{ "algo0" }, Algorithm{ "algo1" }, Algorithm{ "algo1-0" }, Algorithm{ "algo3" }, Algorithm{ "algo2" }, Algorithm{ "algo4" } },
            std::vector{ Property{ "P1" }, Property{ "P1" }, Property{ "P3" }, Property{ "P2" } }, algo_dag,
            algo_equality, prop_equality);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Test traits in namespace utils

struct Comparable
{
  Comparable() = delete;
  Comparable(int) {}
  bool operator<(const Comparable&) const { return true; }
};

struct NotComparable
{
  NotComparable() = delete;
  NotComparable(int){};
};

static_assert(SGraph::utils::has_less_v<int>);
static_assert(SGraph::utils::has_less_v<std::string>);
static_assert(SGraph::utils::has_less_v<Comparable>);
static_assert(!SGraph::utils::has_less_v<NotComparable>);

struct NotStreamConvertible
{};
static_assert(SGraph::utils::is_stream_convertible_v<int>);
static_assert(SGraph::utils::is_stream_convertible_v<std::string>);
static_assert(!SGraph::utils::is_stream_convertible_v<NotStreamConvertible>);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct DefaultType
{};

TEST(DirectedAcyclicGraphTest, VertexComparatorTest) {
  auto i = 1, j = 2;
  auto val = SGraph::GraphBase<int, DefaultType>::m_vertex_less_comparator(std::cref(i), std::cref(j));
  EXPECT_TRUE(val);
  val = SGraph::GraphBase<int, DefaultType>::m_vertex_less_comparator(std::cref(j), std::cref(i));
  EXPECT_FALSE(val);
  Comparable c{ 1 };
  val = SGraph::GraphBase<Comparable, DefaultType>::m_vertex_less_comparator(std::cref(c), std::cref(c));
  EXPECT_TRUE(val); // comparator returns always true
  NotComparable nc{ 1 };
  val = SGraph::GraphBase<NotComparable, DefaultType>::m_vertex_less_comparator(std::cref(nc), std::cref(nc));
  EXPECT_FALSE(val); // compare addresses. Are equal
}

/*---------------------------------------------------------------------------*/

TEST(DirectedAcyclicGraphTest, EdgeComparatorTest) {
  auto i = 1, j = 2;
  auto val = SGraph::GraphBase<DefaultType, int>::m_edge_less_comparator(std::cref(i), std::cref(j));
  EXPECT_TRUE(val);
  val = SGraph::GraphBase<DefaultType, int>::m_edge_less_comparator(std::cref(j), std::cref(i));
  EXPECT_FALSE(val);
  Comparable c{ 1 };
  val = SGraph::GraphBase<DefaultType, Comparable>::m_edge_less_comparator(std::cref(c), std::cref(c));
  EXPECT_TRUE(val); // comparator returns always true
  NotComparable nc{ 1 };
  val = SGraph::GraphBase<DefaultType, NotComparable>::m_edge_less_comparator(std::cref(nc), std::cref(nc));
  EXPECT_FALSE(val); // compare addresses. Are equal
}

/*---------------------------------------------------------------------------*/

TEST(DirectedAcyclicGraphTest, VertexStreamConverterTest) {
  auto stream = SGraph::GraphBase<int, DefaultType>::m_vertex_stream_converter(1);
  EXPECT_EQ(stream, std::string{ "1" });
  auto not_convertible = NotStreamConvertible{};
  stream = SGraph::GraphBase<NotStreamConvertible, DefaultType>::m_vertex_stream_converter(not_convertible);
  std::ostringstream oss;
  oss << &not_convertible;
  EXPECT_EQ(stream, oss.str());
}

/*---------------------------------------------------------------------------*/

TEST(DirectedAcyclicGraphTest, EdgeStreamConverterTest) {
  auto stream = SGraph::GraphBase<DefaultType, std::string>::m_edge_stream_converter("1");
  EXPECT_EQ(stream, "1");
  auto not_convertible = NotStreamConvertible{};
  stream = SGraph::GraphBase<DefaultType, NotStreamConvertible>::m_edge_stream_converter(not_convertible);
  std::ostringstream oss;
  oss << &not_convertible;
  EXPECT_EQ(stream, oss.str());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/