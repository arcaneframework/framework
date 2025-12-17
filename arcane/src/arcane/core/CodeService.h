// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CodeService.h                                               (C) 2000-2025 */
/*                                                                           */
/* Service du code.                                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CODESERVICE_H
#define ARCANE_CORE_CODESERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/List.h"
#include "arcane/utils/UtilsTypes.h"

#include "arcane/core/ICodeService.h"
#include "arcane/core/ServiceBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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

  explicit CodeService(const ServiceBuildInfo& sbi);
  ~CodeService() override;

 public:

  bool parseArgs(StringList&) override { return false; }

  ISubDomain* createAndLoadCase(ISession* session, const SubDomainBuildInfo& sdbi) override;
  void initCase(ISubDomain* sub_domain, bool is_continue) override;
  bool allowExecution() const override;
  StringCollection validExtensions() const override;
  Real lengthUnit() const override { return 1.0; }

 public:

  IServiceInfo* serviceInfo() const override;
  IBase* serviceParent() const override;
  IService* serviceInterface() override { return this; }

 protected:

  void _addExtension(const String& extension);
  IApplication* _application() const;

  virtual void _preInitializeSubDomain(ISubDomain*) {}

 private:

  CodeServicePrivate* m_p = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

