// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MetisWrapper.h                                              (C) 2000-2024 */
/*                                                                           */
/* Wrapper autour des appels de Parmetis.                                    */
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
 * \brief Wrapper autour des appels de Parmetis.
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
   * \brief Simple wrapper autour de la routine ParMetis "ParMETIS_V3_PartKway".
   * 
   * Lorsque gather == true, le graph est regroupe sur 2 processeurs avant appel
   * a ParMETIS_V3_PartKway. Lorsque print_digest == true, la signature des
   * entrees / sorties de ParMETIS_V3_PartKway est affichee.
   */
  int callPartKway(const bool print_digest, const bool gather,
                   idx_t* vtxdist, idx_t* xadj, idx_t* adjncy, idx_t* vwgt,
                   idx_t* adjwgt, idx_t* wgtflag, idx_t* numflag, idx_t* ncon, idx_t* nparts,
                   real_t* tpwgts, real_t* ubvec, idx_t* options, idx_t* edgecut, idx_t* part);

  /*!
   * \brief Simple wrapper autour de la routine ParMetis "ParMETIS_V3_AdaptiveRepart".
   * 
   * Lorsque gather == true, le graph est regroupe sur 2 processeurs avant appel
   * a ParMETIS_V3_AdaptiveRepart. Lorsque print_digest == true, la signature des
   * entrees / sorties de ParMETIS_V3_AdaptiveRepart est affichee.
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
