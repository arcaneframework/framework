// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Parallel.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Namespace for types managing parallelism.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PARALLEL_H
#define ARCANE_CORE_PARALLEL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/ArrayView.h"

#include "arccore/base/RefDeclarations.h"
#include "arccore/message_passing/Request.h"
#include "arccore/message_passing/Communicator.h"
#include "arccore/message_passing/PointToPointMessageInfo.h"
#include "arccore/message_passing/IControlDispatcher.h"

#include "arcane/core/ArcaneTypes.h"

#define ARCANE_BEGIN_NAMESPACE_PARALLEL \
  namespace Parallel \
  {
#define ARCANE_END_NAMESPACE_PARALLEL }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Types of parallelism classes.
 */
namespace Arcane::Parallel
{
class IStat;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Declarations of types and methods used by message exchange mechanisms.
 */
namespace Arcane::MessagePassing
{

/*!
 * \brief Performs a named barrier with name \a name
 *
 * Performs a named barrier \a name using the manager \a pm.
 *
 * All ranks of \a pm block in this barrier and verify that all ranks use the
 * same barrier name. If one of the ranks uses a different name, an exception
 * is raised.
 *
 * This operation allows checking that all ranks use the same barrier, unlike
 * the IParallelMng::barrier() operation.
 *
 * \note Only the first 1024 characters of \a name are used.
 */
extern "C++" ARCANE_CORE_EXPORT void
namedBarrier(IParallelMng* pm, const String& name);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Filters strings common to all ranks of \a pm.
 *
 * Takes an input list \a input_string of strings and returns in
 * \a common_strings those that are common to all ranks of \a pm.
 * The strings returned in \a common_strings are sorted alphabetically.
 */
extern "C++" ARCANE_CORE_EXPORT void
filterCommonStrings(IParallelMng* pm, ConstArrayView<String> input_strings,
                    Array<String>& common_strings);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Writes the date and memory consumed into \a tm.
 *
 * The operation is collective on \a pm and displays the minimum, average,
 * and maximum memory consumed, as well as the ranks of those that consume
 * the least and the most memory.
 */
extern "C++" ARCANE_CORE_EXPORT void
dumpDateAndMemoryUsage(IParallelMng* pm, ITraceMng* tm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream&
operator<<(std::ostream& o, const Parallel::Request prequest)
{
  prequest.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
