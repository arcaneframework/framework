//
// Created by dechaiss on 30/12/2020.
//

#include <iostream>
#include <exception>

#include <string>
#include "gtest/gtest.h"
#include "neo/DirectedGraph.h"


TEST(GraphTest,DirectedGraphTest){
    Neo::DirectedGraph<std::string,std::string> directed_graph{};
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

    EXPECT_THROW(directed_graph.addEdge("f", "h", "fh"),std::runtime_error);

    std::cout << "Edge (e,g) " << *directed_graph.getEdge("e","g");
    std::cout << "Edge (a,b) " << *directed_graph.getEdge("a","b");
    std::cout << "Edge (a,d) " << *directed_graph.getEdge("a","d");
    std::cout << "Edge (b,e) " << *directed_graph.getEdge("b","e");
    std::cout << "Edge (c,e) " << *directed_graph.getEdge("c","e");
    std::cout << "Edge (a,c) " << *directed_graph.getEdge("a","c");
    std::cout << "Edge (e,f) " << *directed_graph.getEdge("e","f");
    std::cout << "Edge (g,h) " << *directed_graph.getEdge("g","h");
    std::cout << "Edge (f,h) " << *directed_graph.getEdge("f","h");
    //
    std::cout << "Edge eg contains nodes " << *directed_graph.getSourceVertex("eg") << " " << *directed_graph.getTargetVertex("eg");
    std::cout << "Edge ab contains nodes " << *directed_graph.getSourceVertex("ab") << " " << *directed_graph.getTargetVertex("ab");

}