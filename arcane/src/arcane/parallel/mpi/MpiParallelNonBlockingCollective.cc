// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiParallelNonBlockingCollective.cc                         (C) 2000-2025 */
/*                                                                           */
/* Implémentation des collectives non bloquantes avec MPI.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/mpi/MpiParallelMng.h"
#include "arcane/parallel/mpi/MpiParallelNonBlockingCollectiveDispatch.h"
#include "arcane/parallel/mpi/MpiParallelNonBlockingCollective.h"

#include "arccore/message_passing_mpi/internal/MpiAdapter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiParallelNonBlockingCollective::
MpiParallelNonBlockingCollective(ITraceMng* tm, IParallelMng* pm, MpiAdapter* adapter)
: ParallelNonBlockingCollectiveDispatcher(pm)
, m_trace_mng(tm)
, m_adapter(adapter)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiParallelNonBlockingCollective::
~MpiParallelNonBlockingCollective()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiParallelNonBlockingCollective::
build()
{
  MpiAdapter* adapter = m_adapter;
  ITraceMng* tm = m_trace_mng;
  auto c = new MpiParallelNonBlockingCollectiveDispatchT<char>(tm, this, adapter);
  auto sc = new MpiParallelNonBlockingCollectiveDispatchT<signed char>(tm, this, adapter);
  auto uc = new MpiParallelNonBlockingCollectiveDispatchT<unsigned char>(tm, this, adapter);
  auto s = new MpiParallelNonBlockingCollectiveDispatchT<short>(tm, this, adapter);
  auto us = new MpiParallelNonBlockingCollectiveDispatchT<unsigned short>(tm, this, adapter);
  auto i = new MpiParallelNonBlockingCollectiveDispatchT<int>(tm, this, adapter);
  auto ui = new MpiParallelNonBlockingCollectiveDispatchT<unsigned int>(tm, this, adapter);
  auto l = new MpiParallelNonBlockingCollectiveDispatchT<long>(tm, this, adapter);
  auto ul = new MpiParallelNonBlockingCollectiveDispatchT<unsigned long>(tm, this, adapter);
  auto ll = new MpiParallelNonBlockingCollectiveDispatchT<long long>(tm, this, adapter);
  auto ull = new MpiParallelNonBlockingCollectiveDispatchT<unsigned long long>(tm, this, adapter);
  auto f = new MpiParallelNonBlockingCollectiveDispatchT<float>(tm, this, adapter);
  auto d = new MpiParallelNonBlockingCollectiveDispatchT<double>(tm, this, adapter);
  auto ld = new MpiParallelNonBlockingCollectiveDispatchT<long double>(tm, this, adapter);
  auto r2 = new MpiParallelNonBlockingCollectiveDispatchT<Real2>(tm, this, adapter);
  auto r3 = new MpiParallelNonBlockingCollectiveDispatchT<Real3>(tm, this, adapter);
  auto r22 = new MpiParallelNonBlockingCollectiveDispatchT<Real2x2>(tm, this, adapter);
  auto r33 = new MpiParallelNonBlockingCollectiveDispatchT<Real3x3>(tm, this, adapter);
  auto hpr = new MpiParallelNonBlockingCollectiveDispatchT<HPReal>(tm, this, adapter);
  _setDispatchers(c, sc, uc, s, us, i, ui, l, ul, ll, ull,
                  f, d, ld, r2, r3, r22, r33, hpr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Parallel::Request MpiParallelNonBlockingCollective::
barrier()
{
  return m_adapter->nonBlockingBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MpiParallelNonBlockingCollective::
hasValidReduceForDerivedType() const
{
  bool is_valid = true;

#if defined(OMPI_MAJOR_VERSION) && defined(OMPI_MINOR_VERSION) && defined(OMPI_RELEASE_VERSION)
  // Pour l'instant toutes les versions connues de OpenMPI (jusqu'à la 1.8.4)
  // ont un bug pour les reduce sur les types dérivés.
  is_valid = false;
#endif

  return is_valid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
