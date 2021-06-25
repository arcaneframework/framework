// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessagePassingMpiGlobal.h                                   (C) 2000-2019 */
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

// Vérifie la version de MPI minimale. Normalement, on ne devrait pas avoir
// de problèmes car cela est vérifié lors de la configuration mais on ne
// sait jamais.
#if !defined(ARCCORE_OS_WIN32)
#if MPI_VERSION < 3 || (MPI_VERSION==3 && MPI_SUBVERSION<1)
#error "MPI_VERSION 3.1 is required. Please disable MPI".
#endif
#endif

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

namespace Arccore::MessagePassing::Mpi
{
class MpiRequest;
class MpiAdapter;
class MpiLock;
class IMpiReduceOperator;
class MpiDatatype;
class MpiMessagePassingMng;
class MpiSerializeMessageList;
class MpiSerializeDispatcher;
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

