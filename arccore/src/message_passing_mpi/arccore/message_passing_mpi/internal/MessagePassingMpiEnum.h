// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessagePassingMpiEnum.h                                     (C) 2000-2025 */
/*                                                                           */
/* Enumeration of different MPI operations.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MESSAGEPASSINGMPIENUM_H
#define ARCCORE_MESSAGEPASSINGMPI_MESSAGEPASSINGMPIENUM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

#include "arccore/collections/CollectionsGlobal.h"

#include "arccore/base/BaseTypes.h"
#include "arccore/base/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Enumeration class for MPI operations.
 */
enum class ARCCORE_MESSAGEPASSINGMPI_EXPORT eMpiName
{
  Bcast = 0,
  Gather = 1,
  Gatherv = 2,
  Allgather = 3,
  Allgatherv = 4,
  Scatterv = 5,
  Alltoall = 6,
  Alltoallv = 7,
  Barrier = 8,
  Reduce = 9,
  Allreduce = 10,
  Scan = 11,
  Sendrecv = 12,
  Isend = 13,
  Send = 14,
  Irecv = 15,
  Recv = 16,
  Test = 17,
  Probe = 18,
  Get_count = 19,
  Wait = 20,
  Waitall = 21,
  Testsome = 22,
  Waitsome = 23,
  NameOffset = 24 // Attention a bien laisser ce champ en dernier (avec sa valeur a jour) si on rajoute des enums !
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Informative structure linked to the enumerations for MPI operations.
 * Provides the name associated with the enum as well as a description of the operation
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiInfo
{
 public:

  // Ctor: we build everything on the fly rather than storing a huge string table
  explicit MpiInfo(eMpiName mpi_operation);
  ~MpiInfo() = default;

  //! Accessor for the name associated with the enum
  const String& name() const;

  //! Accessor for the description associated with the enum
  const String& description() const;

 private:

  String m_name;
  String m_description;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{
using Arcane::MessagePassing::Mpi::eMpiName;
using Arcane::MessagePassing::Mpi::MpiInfo;
} // namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
