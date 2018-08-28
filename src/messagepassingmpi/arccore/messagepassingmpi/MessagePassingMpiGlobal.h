/*---------------------------------------------------------------------------*/
/* MessagePassingMpiGlobal.h                                   (C) 2000-2018 */
/*                                                                           */
/* Définitions globales de la composante 'MessagePassingMpi' de 'Arccore'.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MESSAGEPASSINGMPIGLOBAL_H
#define ARCCORE_MESSAGEPASSINGMPI_MESSAGEPASSINGMPIGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/messagepassing/MessagePassingGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Ces deux macros permettent de s'assurer qu'on ne compile avec le support
// de MpiC++ qui est obsolète
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#endif
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#endif
#include <mpi.h>
#if defined(ARCCORE_OS_WIN32)
// La version de mpi est celle de microsoft. Le proto de MPI_Op doit
// avoir la déclaration __stdcall.
// TODO: verifier avec d'autres MPI sous Windows.
#define ARCCORE_MPIOP_CALL __stdcall
#endif

#ifndef ARCCORE_MPIOP_CALL
#define ARCCORE_MPIOP_CALL
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPONENT_arccore_messagepassingmpi)
#define ARCCORE_MESSAGEPASSINGMPI_EXPORT ARCCORE_EXPORT
#define ARCCORE_MESSAGEPASSINGMPI_EXTERN_TPL
#else
#define ARCCORE_MESSAGEPASSINGMPI_EXPORT ARCCORE_IMPORT
#define ARCCORE_MESSAGEPASSINGMPI_EXTERN_TPL extern
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

namespace MessagePassing
{

namespace Mpi
{
class MpiAdapter;
class MpiLock;
class IMpiReduceOperator;
class MpiDatatype;
class MpiMessagePassingMng;
template<typename DataType>
class StdMpiReduceOperator;
}

}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

