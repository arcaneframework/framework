// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessagePassingMpiEnum.h                                     (C) 2000-2025 */
/*                                                                           */
/* Enumeration des differentes operations MPI.                               */
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
 * \brief Classe enumeration pour les operations MPI.
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
 * \brief Structure informative liee aux enumerationx pour les operations MPI.
 * Donne le nom associe a l'enum ainsi qu'une description de l'operation
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiInfo
{
 public:

  // Ctor : on construit tout a la volee plutot que de stocker une enorme table de string
  explicit MpiInfo(eMpiName mpi_operation);
  ~MpiInfo() = default;

  //! Accesseur sur le nom associe a l'enum
  const String& name() const;

  //! Accesseur sur la description associee a l'enum
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
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
