// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringDictionary.h                                          (C) 2000-2025 */
/*                                                                           */
/* Character string dictionary.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_STRINGDICTIONARY_H
#define ARCCORE_COMMON_STRINGDICTIONARY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Unicode string dictionary.
 *
 * Maintains a list of (key,value) pairs allowing one
 * character string to another. This type of dictionary is
 * used, for example, for translations, in which case the key is
 * the language and the value is the corresponding translation.
 */
class ARCCORE_COMMON_EXPORT StringDictionary
{
 private:

  class Impl; //!< Implementation

 public:

  //! Constructs a dictionary
  StringDictionary();
  //! Constructs a dictionary
  StringDictionary(const StringDictionary& rhs);
  ~StringDictionary(); //!< Releases resources

 public:

  /*! \brief Adds the (key,value) pair to the dictionary.
   *
   * If a value already exists for \a key, it is replaced by
   * the new one.
   */
  void add(const String& key, const String& value);

  /*! \brief Removes the value associated with \a key.
   *
   * If no value was associated with \a key, nothing happens.
   * \return the removed value if there is one.
   */
  String remove(const String& key);

  /*! \brief Returns the value associated with \a key.
   *
   * If no value is associated with \a key, the null string is returned.
   * It is not possible to distinguish between a value
   * corresponding to the null string and a not found value unless
   * \a throw_exception is true, in which case an exception is thrown
   * if there is no value corresponding to \a key.
   */
  String find(const String& key, bool throw_exception = false) const;

  //! Fills \a keys and \a values with the corresponding values from the dictionary
  void fill(StringList& param_names, StringList& values) const;

 private:

  Impl* m_p; //!< Implementation
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
