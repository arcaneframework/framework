// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeLoop.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Boucle en temps.                                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_TIMELOOP_H
#define ARCANE_CORE_TIMELOOP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ITimeLoop.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class TimeLoopPrivate;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Boucle en temps.
 */
class ARCANE_CORE_EXPORT TimeLoop
: public ITimeLoop
{
 public:

  TimeLoop(IApplication* app, const String& name);
  ~TimeLoop() override;

 public:

  virtual void build() override;
  ;

 public:

  IApplication* application() const override;
  String name() const override;
  String title() const override;
  void setTitle(const String& title) override;
  String description() const override;
  void setDescription(const String& description) override;
  StringCollection requiredModulesName() const override;
  void setRequiredModulesName(const StringCollection& names) override;
  StringCollection optionalModulesName() const override;
  void setOptionalModulesName(const StringCollection& names) override;
  TimeLoopEntryPointInfoCollection entryPoints(const String& where) const override;
  void setEntryPoints(const String& where, const TimeLoopEntryPointInfoCollection& calls) override;
  StringCollection userClasses() const override;
  void setUserClasses(const StringCollection& user_classes) override;
  TimeLoopSingletonServiceInfoCollection singletonServices() const override;
  void setSingletonServices(const TimeLoopSingletonServiceInfoCollection& c) override;
  IConfiguration* configuration() override;

 public:

  virtual bool isOldFormat() const;
  virtual void setOldFormat(bool is_old);

 private:

  TimeLoopPrivate* m_p = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

