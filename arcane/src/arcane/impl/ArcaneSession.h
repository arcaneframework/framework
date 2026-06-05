// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneSession.h                                             (C) 2000-2017 */
/*                                                                           */
/* Default implementation of a Session.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_ARCANESESSION_H
#define ARCANE_IMPL_ARCANESESSION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/Session.h"

#include "arcane/core/Directory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IXmlDocumentHolder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Session.
 */
class ARCANE_IMPL_EXPORT ArcaneSession
: public Session
{
 public:

  ArcaneSession(IApplication*);
  virtual ~ArcaneSession();

 public:

  virtual void build();
  virtual void writeExecInfoFile();
  virtual void endSession(int ret_val);
  virtual void setCaseName(String casename);
  virtual void setLogAndErrorFiles(ISubDomain* sd);

 private:

  IXmlDocumentHolder* m_result_doc; //!< Code results
  Directory m_listing_directory;
  Directory m_output_directory;
  String m_case_name;

 private:

  void _checkExecInfoFile();
  void _initSubDomain(ISubDomain* sd);
  void _writeExecInfoFileInit();
  void _writeExecInfoFile(int ret_val);
  void _setLogAndErrorFiles(ISubDomain* sd)
  {
    setLogAndErrorFiles(sd);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
