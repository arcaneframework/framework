// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiTypeDispatcher.cc                                        (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de parallélisme utilisant MPI.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MpiTypeDispatcherImpl.h"

#include "arccore/base/BFloat16.h"
#include "arccore/base/Float16.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class MpiTypeDispatcher<char>;
template class MpiTypeDispatcher<signed char>;
template class MpiTypeDispatcher<unsigned char>;
template class MpiTypeDispatcher<short>;
template class MpiTypeDispatcher<unsigned short>;
template class MpiTypeDispatcher<int>;
template class MpiTypeDispatcher<unsigned int>;
template class MpiTypeDispatcher<long>;
template class MpiTypeDispatcher<unsigned long>;
template class MpiTypeDispatcher<long long>;
template class MpiTypeDispatcher<unsigned long long>;
template class MpiTypeDispatcher<float>;
template class MpiTypeDispatcher<double>;
template class MpiTypeDispatcher<long double>;
template class MpiTypeDispatcher<BFloat16>;
template class MpiTypeDispatcher<Float16>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
