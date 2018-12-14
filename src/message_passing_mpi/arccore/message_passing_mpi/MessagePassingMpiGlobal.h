// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* MessagePassingMpiGlobal.h                                   (C) 2000-2018 */
/*                                                                           */
/* Définitions globales de la composante 'MessagePassingMpi' de 'Arccore'.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MESSAGEPASSINGMPIGLOBAL_H
#define ARCCORE_MESSAGEPASSINGMPI_MESSAGEPASSINGMPIGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"

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

#if defined(ARCCORE_COMPONENT_arccore_message_passing_mpi)
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


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace MpiBuiltIn
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline MPI_Datatype
datatype(char)
{ return MPI_CHAR; }
inline MPI_Datatype
datatype(unsigned char)
{ return MPI_CHAR; }
inline MPI_Datatype
datatype(signed char)
{ return MPI_CHAR; }
inline MPI_Datatype
datatype(short)
{ return MPI_SHORT; }
inline MPI_Datatype
datatype(int)
{ return MPI_INT; }
inline MPI_Datatype
datatype(float)
{ return MPI_FLOAT; }
inline MPI_Datatype
datatype(double)
{ return MPI_DOUBLE; }
inline MPI_Datatype
datatype(long double)
{ return MPI_LONG_DOUBLE; }
inline MPI_Datatype
datatype(long int)
{ return MPI_LONG; }
inline MPI_Datatype
datatype(unsigned short)
{ return MPI_UNSIGNED_SHORT; }
inline MPI_Datatype
datatype(unsigned int)
{ return MPI_UNSIGNED; }
inline MPI_Datatype
datatype(unsigned long)
{ return MPI_UNSIGNED_LONG; }
inline MPI_Datatype
datatype(long long)
{ return MPI_LONG_LONG; }
inline MPI_Datatype
datatype(unsigned long long)
{ return MPI_LONG_LONG; }
}

}

}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

