// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeLoop.h                                                  (C) 2000-2014 */
/*                                                                           */
/* Boucle en temps.                                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_TIMELOOP_H
#define ARCANE_TIMELOOP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ITimeLoop.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

  TimeLoop(IApplication* app,const String& name);
  ~TimeLoop();

 public:

  virtual void build();

 public:

  virtual IApplication* application() const;
  virtual String name() const;
  virtual String title() const;
  virtual void setTitle(const String& title);
  virtual bool isOldFormat() const;
  virtual void setOldFormat(bool is_old);
  virtual String description() const;
  virtual void setDescription(const String& description);
  virtual StringCollection requiredModulesName() const;
  virtual void setRequiredModulesName(const StringCollection & names);
  virtual StringCollection optionalModulesName() const;
  virtual void setOptionalModulesName(const StringCollection & names);
  virtual TimeLoopEntryPointInfoCollection entryPoints(const String& where) const;
  virtual void setEntryPoints(const String& where,const TimeLoopEntryPointInfoCollection& calls);
  virtual StringCollection userClasses() const;
  virtual void setUserClasses(const StringCollection & user_classes);
  virtual TimeLoopSingletonServiceInfoCollection singletonServices() const;
  virtual void setSingletonServices(const TimeLoopSingletonServiceInfoCollection& c);
  virtual IConfiguration* configuration();

 private:

  TimeLoopPrivate* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

