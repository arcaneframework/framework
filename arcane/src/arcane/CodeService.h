// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CodeService.h                                               (C) 2000-2014 */
/*                                                                           */
/* Service du code.                                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CODESERVICE_H
#define ARCANE_CODESERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/List.h"
#include "arcane/utils/UtilsTypes.h"

#include "arcane/ICodeService.h"
#include "arcane/ServiceBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CodeServicePrivate;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe abstraite d'un service de code.
 *
 * La classe dérivée doit implémenter ICodeService::createSession()
 */
class ARCANE_CORE_EXPORT CodeService
: public ICodeService
{
 public:


 public:

  CodeService(const ServiceBuildInfo& sbi);
  virtual ~CodeService();

 public:

  virtual bool parseArgs(StringList&)
    { return false; }

  virtual ISubDomain* createAndLoadCase(ISession* session,const SubDomainBuildInfo& sdbi);
  virtual void initCase(ISubDomain* sub_domain,bool is_continue);
  virtual bool allowExecution() const;
  virtual StringCollection validExtensions() const;
  virtual Real lengthUnit() const { return 1.0; }

 public:

  virtual IServiceInfo* serviceInfo() const;
  virtual IBase* serviceParent() const;
  virtual IService* serviceInterface() { return this; }

 protected:

  void _addExtension(const String& extension);
  IApplication* _application() const;

  virtual void _preInitializeSubDomain(ISubDomain* sd)
  {
    ARCANE_UNUSED(sd);
  }

 private:

  CodeServicePrivate* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

