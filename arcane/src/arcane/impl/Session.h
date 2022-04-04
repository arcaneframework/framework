// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Session.h                                                   (C) 2000-2014 */
/*                                                                           */
/* Classe implémentant une session.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_SESSION_H
#define ARCANE_IMPL_SESSION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/ISession.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IApplication;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Session.
 */
class ARCANE_IMPL_EXPORT Session
: public TraceAccessor
, public ISession
{
 private:

  class Impl;

 public:

  Session(IApplication*);
  virtual ~Session();

 public:
	
  virtual void build();
  virtual void initialize() {}

  virtual IBase* objectParent() const;
  virtual String objectNamespaceURI() const;
  virtual String objectLocalName() const;
  virtual VersionInfo objectVersion() const;

  virtual IServiceMng* serviceMng() const;
  virtual IRessourceMng* ressourceMng() const;
  virtual IApplication* application() const;
  virtual ITraceMng* traceMng() const;
  virtual const String& fileName() const;
  virtual ISubDomain* createSubDomain(const SubDomainBuildInfo& sdbi);
  virtual SubDomainCollection subDomains();
  virtual void doAbort();
  virtual void endSession(int ret_val)
  {
    ARCANE_UNUSED(ret_val);
  }
  virtual void writeExecInfoFile() {}
  virtual bool checkIsValidCaseVersion(const String& version);

 public:
	
  const char* msgClassName() const { return "Session"; }

 protected:
  
  IApplication* _application() const;
  virtual void _initSubDomain(ISubDomain* sd)
  {
    ARCANE_UNUSED(sd);
  }

 private:

  Impl* m_p; //!< Implémentation

 private:

  void _readCaseDocument();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

