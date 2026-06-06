// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISession.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Interface of a session.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISESSION_H
#define ARCANE_CORE_ISESSION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for a case execution session.
 *
 * A session manages the execution of a case in a process.
 *
 * This execution can be distributed across multiple sub-domains using multi-threading.
 */
class ARCANE_CORE_EXPORT ISession
: public IBase
{
 public:

  virtual ~ISession() = default; //!< Frees resources

 public:

  //! Application
  virtual IApplication* application() const = 0;

  /*!
    \brief Creates a sub-domain with the parameters contained in \a sdbi.
   
    The created sub-domain is added to the list of sub-domains of
    the session. The document containing the dataset is opened
    and its XML validity is checked, but the options of the services
    and modules are not read.
  */
  virtual ISubDomain* createSubDomain(const SubDomainBuildInfo& sdbi) = 0;

  //! Ends the session with the return code ret_val
  virtual void endSession(int ret_val) = 0;

  //! List of sub-domains of the session
  virtual SubDomainCollection subDomains() = 0;

  //! Performs an abort
  virtual void doAbort() = 0;

  /*!
   * \brief Checks if the dataset version \a version is valid.
   *
   * \retval true if the version is valid
   * \retval false otherwise   
   */
  virtual bool checkIsValidCaseVersion(const String& version) = 0;

  //! Writes the execution information file
  virtual void writeExecInfoFile() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
