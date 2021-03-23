// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AutoDestroyUserData.h                                       (C) 2000-2012 */
/*                                                                           */
/* UserData s'auto-détruisant une fois détaché.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_AUTODESTROYUSERDATA_H
#define ARCANE_UTILS_AUTODESTROYUSERDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IUserData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T>
class DeleteOnDestroyBehaviour
{
 public:
  static void destroy(T* t){ delete t; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief UserData s'auto-détruisant une fois détaché.
 * \ingroup Core
 *
 * Une instance de cette classe doit être allouée via new()
 * et est détruite automatiquement ainsi que sa donnée associée lorsqu'elle
 * est détachée d'un IUserDataList via IUserDataList::removeData().
 *
 * Par défaut, elle appelle l'opérateur delete pour sa donnée
 * mais il est possible de changer son comportement via le
 * paramètre template DestroyBehaviour.
 */
template<typename T,typename DestroyBehaviour = DeleteOnDestroyBehaviour<T> >
class AutoDestroyUserData
: public IUserData
{
 public:
  
  AutoDestroyUserData(T* adata): m_data(adata){}
 private:
  ~AutoDestroyUserData()
  {
  }

 public:

  virtual void notifyAttach(){}

  virtual void notifyDetach()
  {
    DestroyBehaviour::destroy(m_data);
    m_data = 0;
    delete this;
  }
  
  T* data() { return m_data; }

 private:
  T* m_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

