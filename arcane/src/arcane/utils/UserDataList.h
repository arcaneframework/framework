// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UserDataList.h                                              (C) 2000-2018 */
/*                                                                           */
/* Gère une liste de données utilisateurs.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_USERDATALIST_H
#define ARCANE_UTILS_USERDATALIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IUserDataList.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IUserData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère une liste de données utilisateurs.
 * \ingroup Core
 */
class ARCANE_UTILS_EXPORT UserDataList
: public IUserDataList
{
 public:

  class Impl;

 public:
	
  UserDataList();
  //! Libère les ressources
  ~UserDataList();

 public:

  virtual void setData(const String& name,IUserData* ud);
  virtual IUserData* data(const String& name,bool allow_null=false) const;
  virtual void removeData(const String& name,bool allow_null=false);
  virtual void clear();

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

