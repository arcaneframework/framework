// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IUserData.h                                                 (C) 2000-2012 */
/*                                                                           */
/* Interface for user data attached to another object.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IUSERDATA_H
#define ARCANE_UTILS_IUSERDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for user data attached to another object.
 * \ingroup Core
 */
class ARCANE_UTILS_EXPORT IUserData
{
 public:
	
  //! Releases resources
  virtual ~IUserData(){}

 public:

  //! Method executed when the instance is attached.
  virtual void notifyAttach() =0;

  //! Method executed when the instance is detached.
  virtual void notifyDetach() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
