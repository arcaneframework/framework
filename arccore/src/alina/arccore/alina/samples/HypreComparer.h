// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HypreComparer.h                                             (C) 2000-2026 */
/*                                                                           */
/* Utilitary class to use Hypre as a solver for sample matrix.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_SAMPLES_HYPRECOMPARER_H
#define ARCCORE_ALINA_SAMPLES_HYPRECOMPARER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/alina/AlinaGlobal.h"

#include <vector>
#include <cstddef>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HypreComparer
{
 public:

  HypreComparer(bool do_mpi_init_and_finalize)
  : m_do_mpi_init_and_finalize(do_mpi_init_and_finalize)
  {
  }
  ~HypreComparer();

 public:

  void solve(int nb_row,
             std::vector<ptrdiff_t> const& _ptr,
             std::vector<ptrdiff_t> const& _col,
             std::vector<double> const& _val,
             std::vector<double> const& _rhs,
             std::vector<double>& _x,
             int argc, char* argv[]);

 private:

  bool m_do_mpi_init_and_finalize = false;
  bool m_need_finalize = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
