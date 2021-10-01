//
// Created by dechaiss on 30/12/2020.
//

#include <iostream>

#include <string>

#include <gtest/gtest.h>

#include "sgraph/DirectedGraph.h"
#include "sgraph/DirectedAcyclicGraph.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/



//----------------------------------------------------------

struct Property{std::string name;};

struct Algorithm{std::string name;};

std::ostream& operator<<(std::ostream& os, Algorithm* algo){ os << algo->name; return os;}

std::ostream& operator<<(std::ostream& os, Property* prop){ os << prop->name; return os;}

TEST(DirectedGraphTest,UnitTest) {

	
	SGraph::DirectedGraph<Property*,Algorithm*> directed_graph{};
	auto prop1 = Property{"prop_in"};
	auto prop2 = Property{"prop_out"};
	auto prop3 = Property{"prop_final"};
	auto algo1 = Algorithm{"algo1"};
	auto algo2 = Algorithm{"algo2"};
	directed_graph.addEdge(&prop2,&prop3,&algo2);
	directed_graph.addEdge(&prop1,&prop2,&algo1);
	directed_graph.print();
}


TEST (DirectedGraphTest,StringGraphTest) {
	
	SGraph::DirectedGraph<std::string,std::string> directed_graph{};
    directed_graph.addEdge("a", "b", "ab");
    directed_graph.addEdge("e", "g", "eg");
    
	std::cout << "getEdge (\"e\",\"g\") " << *directed_graph.getEdge("e","g") << std::endl;
	std::cout << "getEdge (\"a\",\"b\") " << *directed_graph.getEdge("a","b") << std::endl;
	
	
	std::cout << "sourceVertex(\"eg\") " <<*directed_graph.getSourceVertex("eg") << std::endl;
	std::cout << "targetVertex(\"eg\") " <<*directed_graph.getTargetVertex("eg") << std::endl;
	
	directed_graph.print();
	
	//directed_graph.addEdge("a", "b", "ab");
    directed_graph.addEdge("a", "d", "ad");
    directed_graph.addEdge("b", "e", "be");
    directed_graph.addEdge("c", "e", "ce");
    directed_graph.addEdge("a", "c", "ac");
    directed_graph.addEdge("e", "f", "ef");
    directed_graph.addEdge("g", "h", "gh");
    directed_graph.addEdge("f", "h", "fh");

    //EXPECT_THROW(directed_graph.addEdge("f", "h", "fh"),std::runtime_error);

   
    std::cout << "Edge (a,b) " << *directed_graph.getEdge("a","b") <<std::endl;
    std::cout << "Edge (a,d) " << *directed_graph.getEdge("a","d") <<std::endl;
    std::cout << "Edge (b,e) " << *directed_graph.getEdge("b","e") <<std::endl;
    std::cout << "Edge (c,e) " << *directed_graph.getEdge("c","e") <<std::endl;
    std::cout << "Edge (a,c) " << *directed_graph.getEdge("a","c") <<std::endl;
    std::cout << "Edge (e,f) " << *directed_graph.getEdge("e","f") <<std::endl;
    std::cout << "Edge (g,h) " << *directed_graph.getEdge("g","h") <<std::endl;
    std::cout << "Edge (f,h) " << *directed_graph.getEdge("f","h") <<std::endl;
    //
    std::cout << "Edge eg contains nodes " << *directed_graph.getSourceVertex("eg") << " " << *directed_graph.getTargetVertex("eg") << std::endl;
    std::cout << "Edge ab contains nodes " << *directed_graph.getSourceVertex("ab") << " " << *directed_graph.getTargetVertex("ab") << std::endl;
	}

//-----------------------------------

TEST (DirectedAcyclicGraphTest,UnitTest)
{
  // Same code base as DirectedGraphT, the topological sort is added
  SGraph::DirectedAcyclicGraph<std::string,std::string> dag{};
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
  for (const std::string& sorted_vertex : dag.topologicalSort()) // pas cool, si on met auto on a un ref_wrapper et donc ça plante...todo...
    {
      std::cout << "Sorted Graph has vertex " << sorted_vertex << std::endl;
    }

  // Print topologically oredered graph in reverse order
   for (const std::string& sorted_vertex : dag.topologicalSort().reverseOrder())
    {
       std::cout << "Reverse order sorted Graph has vertex " << sorted_vertex << std::endl;
    }

  // Print Spanning tree (arbre couvrant)
  for (const std::string& edge_tree : dag.spanningTree())
    {
      std::cout << "Spanning tree has edge " << edge_tree << std::endl;
    }

  // Print Spanning tree (arbre couvrant) in reverse order
  for (const std::string& edge_tree : dag.spanningTree().reverseOrder())
    {
      std::cout << "Reverse order spanning tree has edge " << edge_tree << std::endl;
    }

  // add edge and check impact
  dag.addEdge("a","dprime","adprime");
  dag.addEdge("dprime","g","dprimeg");
  dag.addEdge("h","i","hi");

  for (const std::string& sorted_vertex : dag.topologicalSort())
    {
      std::cout << "Sorted Graph has vertex " << sorted_vertex << std::endl;
    }

  // Print Spanning tree (arbre couvrant)
  for (const std::string& edge_tree : dag.spanningTree())
    {
      std::cout << "Spanning tree has edge " << edge_tree << std::endl;
    }

  // Corrupt the dag inserting a cycle (topologicalSort() and print() will throw runtime_error)
  dag.addEdge("g", "a","ga");
  dag.addEdge("b", "g","bg");

  if (!dag.hasCycle()) throw std::runtime_error{"Error, cycles are not detected in DAG."};

  // The graph is now corrupted...

}
//---------------------------------------------