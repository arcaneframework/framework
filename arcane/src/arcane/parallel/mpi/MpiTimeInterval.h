// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiTimeInterval.h                                           (C) 2000-2022 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_MPITIMEINTERVAL_H
#define ARCANE_PARALLEL_MPI_MPITIMEINTERVAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/mpi/ArcaneMpi.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiTimeInterval
{
 public:

  MpiTimeInterval(double* cumulative_value)
  : m_cumulative_value(cumulative_value)
  {
    m_begin_time = _getTime();
  }
  ~MpiTimeInterval()
  {
    double end_time = _getTime();
    *m_cumulative_value += (end_time - m_begin_time);
  }

 private:

  inline double _getTime()
  {
    return MPI_Wtime();
  }

 private:

  double* m_cumulative_value;
  double m_begin_time = 0.0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
