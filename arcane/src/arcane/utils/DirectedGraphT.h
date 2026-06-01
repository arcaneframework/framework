// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DirectedGraphT.h                                            (C) 2000-2017 */
/*                                                                           */
/* Comment on file content.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DIRECTEDGRAPHT_H_
#define ARCANE_DIRECTEDGRAPHT_H_
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/GraphBaseT.h"
#include "arcane/utils/ITraceMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*! Template class for DirectedGraph.
 *  VertexType must implement a less comparison operator.
 *
 */
template <class VertexType, class EdgeType>
class DirectedGraphT
: public GraphBaseT<VertexType, EdgeType>
{
 public:

  /** Constructor of the class */
  DirectedGraphT(ITraceMng* trace_mng)
  : GraphBaseT<VertexType, EdgeType>(trace_mng)
  {}

  /** Destructor of the class */
  virtual ~DirectedGraphT() {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
