// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDirectory.h                                                (C) 2000-2025 */
/*                                                                           */
/* Directory management.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IDIRECTORY_H
#define ARCANE_CORE_IDIRECTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a class managing a directory.
 */
class ARCANE_CORE_EXPORT IDirectory
{
 public:

  virtual ~IDirectory() = default; //!< Frees the resources

 public:

  /*!
   * \brief Creates the directory.
   * \retval true in case of failure,
   * \retval false in case of success or if the directory already exists.
   */
  virtual bool createDirectory() const = 0;

  //! Returns the path of the directory
  virtual String path() const = 0;

  //! Returns the full path of the file \a file_name in the directory
  virtual String file(const String& file_name) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
