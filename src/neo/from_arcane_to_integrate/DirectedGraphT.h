// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* DirectedGraph.h                                             (C) 2000-2017 */
/*                                                                           */
/* Comment on file content.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DIRECTEDGRAPHT_H_ 
#define ARCANE_DIRECTEDGRAPHT_H_ 
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "neo/Utils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Neo{
    namespace utils{

    /*---------------------------------------------------------------------------*/
    /*---------------------------------------------------------------------------*/
    /*! Template class for DirectedGraph.
     *  VertexType must implement a less comparison operator.
     *
     */
    template <class VertexType, class EdgeType>
    class DirectedGraphT
    : public GraphBaseT<VertexType,EdgeType>
    {
    public:

      /** Constructeur de la classe */
      DirectedGraphT(ITraceMng* trace_mng)
      : GraphBaseT<VertexType,EdgeType>(trace_mng) {}

      /** Destructeur de la classe */
      virtual ~DirectedGraphT() {}
    };

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* DIRECTEDGRAPHT_H_ */
