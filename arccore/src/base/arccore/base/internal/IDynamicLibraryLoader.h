// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDynamicLibraryLoader.h                                     (C) 2000-2025 */
/*                                                                           */
/* Interface of a dynamic library loader.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_INTERNAL_IDYNAMICLIBRARYLOADER_H
#define ARCCORE_BASE_INTERNAL_IDYNAMICLIBRARYLOADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a dynamic library.
 */
class ARCCORE_BASE_EXPORT IDynamicLibrary
{
 public:

  virtual ~IDynamicLibrary() = default; //!< Releases resources
 public:

  /*!
   * \brief Closes the dynamic library.
   *
   * It should no longer be used after closing and the instance can
   * be destroyed via the delete operator.
   */
  virtual void close() = 0;

  /*!
   * \brief Returns the address of the symbol named \a symbol_name.
   *
   * If \a is_found is not null, it returns the boolean indicating
   * whether the symbol was found.
   */
  virtual void* getSymbolAddress(const String& symbol_name, bool* is_found) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal.
 *
 * \brief Interface of a dynamic library loader.
 */
class ARCCORE_BASE_EXPORT IDynamicLibraryLoader
{
 public:

  virtual ~IDynamicLibraryLoader() = default; //!< Releases resources

 public:

  virtual void build() = 0;

 public:

  /*!
   * \brief Loads a dynamic library.
   *
   * Loads the library named \a name which is located in the directory
   * \a directory. Returns a null pointer if the library cannot
   * be loaded. \a name must be a name without prefix and without machine-dependent extension.
   * For example, on linux, if the library is
   * libtoto.so, \a name must be \a toto.
   */
  virtual IDynamicLibrary* open(const String& directory, const String& name) = 0;

  //! Closes all libraries opened via \a open()
  virtual void closeLibraries() = 0;

 public:

  /*!
   * \brief Service used for dynamically loading libraries.
   */
  static IDynamicLibraryLoader* getDefault();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
