// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* MpiTypeDispatcher.cc                                        (C) 2000-2018 */
/*                                                                           */
/* Gestionnaire de parallélisme utilisant MPI.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MpiTypeDispatcherImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
namespace MessagePassing
{
namespace Mpi
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Mpi
} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
