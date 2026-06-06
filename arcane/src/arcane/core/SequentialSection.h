// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SequentialSection.h                                         (C) 2000-2025 */
/*                                                                           */
/* Section of code to be executed sequentially.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SEQUENTIALSECTION_H
#define ARCANE_CORE_SEQUENTIALSECTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ParallelFatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Section of code to be executed sequentially.
 *
 * An instance of this class allows a part of the code to run
 * as if the code were sequential. The code within the lifetime
 * of this object is first executed on proc 0, and then, if everything is
 * okay on the others. This allows checking it once when the code executed
 * is the same everywhere (for example, reading the dataset) and displaying
 * messages only once in case of an error.
 *
 * Since potential errors are only displayed by a single
 * subdomain, this class should only be used when you are certain that all
 * subdomains perform the same processing, otherwise errors will not be
 * recognized.
 *
 * Furthermore, since all subdomains block until the first
 * subdomain has finished executing the code, you must not call the
 * parallelism manager in this section.
 *
 * In case of an error, an exception of type ExParallelFatalError is
 * sent in the destructor.
 * \code
 * {
 *   SequentialSection ss(pm);
 *   ... // Code executed sequentially.
 *   ss.setError(true);
 * }
 * \endcode
 *
 */
class ARCANE_CORE_EXPORT SequentialSection
{
 public:

  explicit SequentialSection(IParallelMng*);
  explicit SequentialSection(ISubDomain*);
  ~SequentialSection() ARCANE_NOEXCEPT_FALSE;

 public:

  void setError(bool is_error);

 private:

  IParallelMng* m_parallel_mng = nullptr;
  bool m_has_error = false;

  void _init();
  void _sendError();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
