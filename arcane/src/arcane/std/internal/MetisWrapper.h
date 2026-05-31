// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MetisWrapper.h                                              (C) 2000-2024 */
/*                                                                           */
/* Wrapper around Parmetis calls.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_METISWRAPPER
#define ARCANE_STD_METISWRAPPER
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include <parmetis.h>
#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class MetisGraphView;
class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Wrapper around Parmetis calls.
 */
class MetisWrapper
: public TraceAccessor
{
 private:

  using MetisCall = std::function<int(IParallelMng* pm, MetisGraphView graph,
                                      ArrayView<idx_t> vtxdist)>;

 public:

  explicit MetisWrapper(IParallelMng* pm);

 public:

  /*!
   * \brief Simple wrapper around the ParMetis routine "ParMETIS_V3_PartKway".
   * 
   * When gather == true, the graph is grouped on 2 processors before calling
   * ParMETIS_V3_PartKway. When print_digest == true, the signature of the
   * inputs/outputs of ParMETIS_V3_PartKway is displayed.
   */
  int callPartKway(const bool print_digest, const bool gather,
                   idx_t* vtxdist, idx_t* xadj, idx_t* adjncy, idx_t* vwgt,
                   idx_t* adjwgt, idx_t* wgtflag, idx_t* numflag, idx_t* ncon, idx_t* nparts,
                   real_t* tpwgts, real_t* ubvec, idx_t* options, idx_t* edgecut, idx_t* part);

  /*!
   * \brief Simple wrapper around the ParMetis routine "ParMETIS_V3_AdaptiveRepart".
   * 
   * When gather == true, the graph is grouped on 2 processors before calling
   * ParMETIS_V3_AdaptiveRepart. When print_digest == true, the signature of the
   * inputs/outputs of ParMETIS_V3_AdaptiveRepart is displayed.
   */
  int callAdaptiveRepart(const bool print_digest, const bool gather,
                         idx_t* vtxdist, idx_t* xadj, idx_t* adjncy, idx_t* vwgt,
                         idx_t* vsize, idx_t* adjwgt, idx_t* wgtflag, idx_t* numflag, idx_t* ncon,
                         idx_t* nparts, real_t* tpwgts, real_t* ubvec, real_t* ipc2redist,
                         idx_t* options, idx_t* edgecut, idx_t* part);

 private:

  IParallelMng* m_parallel_mng = nullptr;

 private:

  int _callMetisWith2Processors(const Int32 ncon, const bool need_part,
                                ConstArrayView<idx_t> vtxdist, MetisGraphView my_graph,
                                MetisCall& metis);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
