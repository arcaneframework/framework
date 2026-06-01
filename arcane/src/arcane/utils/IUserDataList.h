// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IUserDataList.h                                             (C) 2000-2018 */
/*                                                                           */
/* Interface of a list that manages user data.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IUSERDATALIST_H
#define ARCANE_UTILS_IUSERDATALIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IUserData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a list that manages user data.
 * \ingroup Core
 */
class ARCANE_UTILS_EXPORT IUserDataList
{
 public:
	
  //! Frees resources
  virtual ~IUserDataList(){}

 public:

  /*!
   * \brief Sets the user data associated with the name \a name.
   *
   * No data should already be associated with \a name, otherwise an
   * exception is thrown.
   */  
  virtual void setData(const String& name,IUserData* ud) =0;

  /*!
   * \brief Data associated with \a name.
   *
   * An exception is thrown if \a allow_null is \a false and no
   * data is associated with \a name. If \a allow_null is \a true and
   * no data is associated, a null pointer is returned.
   */
  virtual IUserData* data(const String& name,bool allow_null=false) const =0;

  /*!
   * \brief Removes the data associated with the name \a name.
   *
   * An exception is thrown if \a allow_null is \a false and no
   * data is associated with \a name.
   */
  virtual void removeData(const String& name,bool allow_null=false) =0;

  /*!
   * \brief Removes all user data.
   *
   * This is equivalent to calling removeData() for all user data.
   */  
  virtual void clear() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
