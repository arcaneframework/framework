// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AutoDestroyUserData.h                                       (C) 2000-2012 */
/*                                                                           */
/* UserData that self-destructs once detached.                               */
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
 * \brief UserData that self-destructs once detached.
 * \ingroup Core
 *
 * An instance of this class must be allocated via new()
 * and is automatically destroyed along with its associated data when it
 * is detached from an IUserDataList via IUserDataList::removeData().
 *
 * By default, it calls the delete operator for its data
 * but it is possible to change its behavior via the
 * DestroyBehaviour template parameter.
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
