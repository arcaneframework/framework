//
// Created by dechaiss on 30/12/2020.
//

#ifndef NEO_DIRECTEDGRAPH_H
#define NEO_DIRECTEDGRAPH_H

#include "neo/GraphBase.h"

namespace Neo {

    /*---------------------------------------------------------------------------*/
    /*---------------------------------------------------------------------------*/
    /*! Template class for DirectedGraph.
     *  VertexType must implement a less comparison operator.
     *
     */
    template<class VertexType, class EdgeType>
    class DirectedGraph
            : public GraphBase<VertexType, EdgeType> {
    public:

        /** Constructeur de la classe */
        DirectedGraph() : GraphBase<VertexType, EdgeType>() {}

        /** Destructeur de la classe */
        virtual ~DirectedGraph() {}
    };


}// namespace Neo


#endif //NEO_DIRECTEDGRAPH_H
