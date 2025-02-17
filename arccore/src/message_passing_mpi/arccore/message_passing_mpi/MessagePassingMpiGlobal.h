﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessagePassingMpiGlobal.h                                   (C) 2000-2025 */
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
#if MPI_VERSION < 3 || (MPI_VERSION == 3 && MPI_SUBVERSION < 1)
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

namespace Arcane::MessagePassing::Mpi
{
class MpiRequest;
class MpiAdapter;
class MpiLock;
class IMpiReduceOperator;
class IMpiProfiling;
class NoMpiProfiling;
class MpiDatatype;
class MpiMessagePassingMng;
class MpiSerializeMessageList;
class MpiSerializeDispatcher;
class StandaloneMpiMessagePassingMng;
class MpiSerializeMessageList;
template <typename DataType>
class StdMpiReduceOperator;
template <typename Type>
class MpiTypeDispatcher;
} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi::MpiBuiltIn
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline MPI_Datatype
datatype(char)
{
  return MPI_CHAR;
}
inline MPI_Datatype
datatype(unsigned char)
{
  return MPI_CHAR;
}
inline MPI_Datatype
datatype(signed char)
{
  return MPI_CHAR;
}
inline MPI_Datatype
datatype(short)
{
  return MPI_SHORT;
}
inline MPI_Datatype
datatype(int)
{
  return MPI_INT;
}
inline MPI_Datatype
datatype(float)
{
  return MPI_FLOAT;
}
inline MPI_Datatype
datatype(double)
{
  return MPI_DOUBLE;
}
inline MPI_Datatype
datatype(long double)
{
  return MPI_LONG_DOUBLE;
}
inline MPI_Datatype
datatype(long int)
{
  return MPI_LONG;
}
inline MPI_Datatype
datatype(unsigned short)
{
  return MPI_UNSIGNED_SHORT;
}
inline MPI_Datatype
datatype(unsigned int)
{
  return MPI_UNSIGNED;
}
inline MPI_Datatype
datatype(unsigned long)
{
  return MPI_UNSIGNED_LONG;
}
inline MPI_Datatype
datatype(long long)
{
  return MPI_LONG_LONG;
}
inline MPI_Datatype
datatype(unsigned long long)
{
  return MPI_LONG_LONG;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi::MpiBuiltIn

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{
// Pour compatibilité avec Alien
using Arcane::MessagePassing::internal::BasicSerializeMessage;

using Arcane::MessagePassing::Mpi::IMpiProfiling;
using Arcane::MessagePassing::Mpi::IMpiReduceOperator;
using Arcane::MessagePassing::Mpi::MpiAdapter;
using Arcane::MessagePassing::Mpi::MpiDatatype;
using Arcane::MessagePassing::Mpi::MpiLock;
using Arcane::MessagePassing::Mpi::MpiMessagePassingMng;
using Arcane::MessagePassing::Mpi::MpiRequest;
using Arcane::MessagePassing::Mpi::MpiSerializeDispatcher;
using Arcane::MessagePassing::Mpi::MpiSerializeMessageList;
using Arcane::MessagePassing::Mpi::MpiTypeDispatcher;
using Arcane::MessagePassing::Mpi::NoMpiProfiling;
using Arcane::MessagePassing::Mpi::StandaloneMpiMessagePassingMng;
using Arcane::MessagePassing::Mpi::StdMpiReduceOperator;
namespace MpiBuiltIn = Arcane::MessagePassing::Mpi::MpiBuiltIn;
}; // namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
