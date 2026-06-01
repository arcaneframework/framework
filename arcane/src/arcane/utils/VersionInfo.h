// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VersionInfo.h                                               (C) 2000-2018 */
/*                                                                           */
/* Information about an object's version.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_VERSIONINFO_H
#define ARCANE_UTILS_VERSIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information about a version.
 *
 This class contains information about an object's version.
 The version number comprises 3 integer values:

 \arg the major version number,
 \arg the minor version number,
 \arg the patch version number,

 The major version number corresponds to a fundamental evolution
 of the object. The minor version number corresponds to less significant evolutions. A major or minor version evolution
 implies that binary compatibility is not maintained.

 \note the sub-version number is no longer used.
 */
class ARCANE_UTILS_EXPORT VersionInfo
{
 public:

  //! Constructs a null version
  VersionInfo();

  //! Constructs a version information
  VersionInfo(int vmajor,int vminor,int vpatch);

  /*! \brief Constructs a version information
   * \a version_str must be in the format "M.m.p.b" where M is the major version,
   * \m is the minor version, p is the patch number, and b is the beta number.
   */
  VersionInfo(const Arccore::String& version_str);

 public:
	
  //! Returns the major version number
  int versionMajor() const { return m_major; }

  //! Returns the minor version number
  int versionMinor() const { return m_minor; }

  //! Returns the patch version number
  int versionPatch() const { return m_patch; }

  //! Version number in string format
  String versionAsString() const;

 public:

  // Prints the version numbers to the stream o
  void write(std::ostream& o) const;

 private:

  int m_major; //!< Major version number
  int m_minor; //!< Minor version number
  int m_patch; //!< Patch version number
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT std::ostream&
operator<<(std::ostream& o,const VersionInfo& vi);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
