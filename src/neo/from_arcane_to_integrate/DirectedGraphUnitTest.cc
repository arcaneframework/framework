// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* DirectedGraphUnitTest.cc                                    (C) 2000-2017 */
/*                                                                           */
/* Unit test of directed graphes based on GraphBaseT class                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/BasicUnitTest.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/DirectedGraphUnitTest_axl.h"

#include "arcane/utils/DirectedAcyclicGraphT.h"
#include "arcane/utils/DirectedGraphT.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test des Ios
 */
class DirectedGraphUnitTest
  : public ArcaneDirectedGraphUnitTestObject
{
public:

  /** Constructeur de la classe */
  DirectedGraphUnitTest(const Arcane::ServiceBuildInfo & sbi)
    : ArcaneDirectedGraphUnitTestObject(sbi) {}

  /** Destructeur de la classe */
  ~DirectedGraphUnitTest() {}

public:

  virtual void initializeTest();
  virtual void executeTest();

private:
  void _testDirectedAcyclicGraph();
  void _testDirectedGraph();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_DIRECTEDGRAPHUNITTEST(DirectedGraphUnitTest,DirectedGraphUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DirectedGraphUnitTest::
initializeTest()
{
  info() << "[DirectedGraphUnitTest] initializeTest";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DirectedGraphUnitTest::
executeTest()
{
  info() << "[DirectedGraphUnitTest] executeTest";
  _testDirectedGraph();
  _testDirectedAcyclicGraph();

}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DirectedGraphUnitTest::
_testDirectedGraph()
{
  DirectedGraphT<String,String> directed_graph(traceMng());
    directed_graph.addEdge("a", "h", "ah");
    directed_graph.addEdge("e", "g", "eg");
    directed_graph.addEdge("a", "b", "ab");
    directed_graph.addEdge("a", "d", "ad");
    directed_graph.addEdge("b", "e", "be");
    directed_graph.addEdge("c", "e", "ce");
    directed_graph.addEdge("a", "c", "ac");
    directed_graph.addEdge("e", "f", "ef");
    directed_graph.addEdge("g", "h", "gh");
    directed_graph.addEdge("f", "h", "fh");
    bool test_add_edge_ok = false;
    try {
        directed_graph.addEdge("f", "h", "fh"); // Must throw fatal, otherwise the test fails
    } catch (FatalErrorException& e) {
        test_add_edge_ok = true;
    }
    if (!test_add_edge_ok) fatal() << "Cannot insert twice an edge between the same nodes";

    info() << "Edge (e,g) " << *directed_graph.getEdge("e","g");
    info() << "Edge (a,b) " << *directed_graph.getEdge("a","b");
    info() << "Edge (a,d) " << *directed_graph.getEdge("a","d");
    info() << "Edge (b,e) " << *directed_graph.getEdge("b","e");
    info() << "Edge (c,e) " << *directed_graph.getEdge("c","e");
    info() << "Edge (a,c) " << *directed_graph.getEdge("a","c");
    info() << "Edge (e,f) " << *directed_graph.getEdge("e","f");
    info() << "Edge (g,h) " << *directed_graph.getEdge("g","h");
    info() << "Edge (f,h) " << *directed_graph.getEdge("f","h");
  //
    info() << "Edge eg contains nodes " << *directed_graph.getSourceVertex("eg") << " " << *directed_graph.getTargetVertex("eg");
    info() << "Edge ab contains nodes " << *directed_graph.getSourceVertex("ab") << " " << *directed_graph.getTargetVertex("ab");

    // Iterate over vertices or edges

    for (String& vertex : directed_graph.vertices())
      {
        info() << "Graph has vertex " << vertex;
      }

    for (String& edge : directed_graph.edges())
      {
        info() << "Graph has edge " << edge;
      }

    for (const String& edge : directed_graph.inEdges("h"))
      {
        info() << "Vertex a has incoming edge " << edge;
      }

    for (const String& edge : directed_graph.outEdges("a"))
      {
        info() << "Vertex a has outcoming edge " << edge;
      }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DirectedGraphUnitTest::
_testDirectedAcyclicGraph()
{
  // Same code base as DirectedGraphT, the topological sort is added
  DirectedAcyclicGraphT<String,String> dag(traceMng());
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

  dag.print();

  // Print topologically oredered graph
  for (const String& sorted_vertex : dag.topologicalSort())
    {
      info() << "Sorted Graph has vertex " << sorted_vertex;
    }

  // Print topologically oredered graph in reverse order
   for (const String& sorted_vertex : dag.topologicalSort().reverseOrder())
    {
       info() << "Reverse order sorted Graph has vertex " << sorted_vertex;
    }

  // Print Spanning tree (arbre couvrant)
  for (const String& edge_tree : dag.spanningTree())
    {
      info() << "Spanning tree has edge " << edge_tree;
    }

  // Print Spanning tree (arbre couvrant) in reverse order
  for (const String& edge_tree : dag.spanningTree().reverseOrder())
    {
      info() << "Reverse order spanning tree has edge " << edge_tree;
    }

  // add edge and check impact
  dag.addEdge("a","dprime","adprime");
  dag.addEdge("dprime","g","dprimeg");
  dag.addEdge("h","i","hi");

  for (const String& sorted_vertex : dag.topologicalSort())
    {
      info() << "Sorted Graph has vertex " << sorted_vertex;
    }

  // Print Spanning tree (arbre couvrant)
  for (const String& edge_tree : dag.spanningTree())
    {
      info() << "Spanning tree has edge " << edge_tree;
    }

  // Corrupt the dag inserting a cycle (topologicalSort() and print() will throw FatalErrorException)
  dag.addEdge("g", "a","ga");
  dag.addEdge("b", "g","bg");

  if (!dag.hasCycle()) fatal() << "Error, cycles are not detected in DAG.";

  // The graph is now corrupted...

}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
