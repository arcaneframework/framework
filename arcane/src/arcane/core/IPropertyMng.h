// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPropertyMng.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface of the property manager.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPROPERTYMNG_H
#define ARCANE_CORE_IPROPERTYMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Properties;
class PropertiesImpl;
class IObservable;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of the property manager.
 */
class IPropertyMng
{
 public:

  virtual ~IPropertyMng() {} //!< Frees the resources.

 public:

  virtual ITraceMng* traceMng() const = 0;

 public:

  /*!
   * \internal
   * \brief Retrieves the list of properties by full name \a full_name.
   *
   * This method must only be called by the Properties class.
   * To retrieve an instance, the Properties constructor must be used.
   */
  virtual PropertiesImpl* getPropertiesImpl(const String& full_name) = 0;

  /*!
   * \internal
   * \brief Registers the properties referenced by \a p.
   */
  virtual void registerProperties(const Properties& p) = 0;

  //! Deletes the properties referenced by \a p
  virtual void destroyProperties(const Properties& p) = 0;

  //! Performs serialization
  virtual void serialize(ISerializer* serializer) = 0;

  //! Serializes property information into \a bytes.
  virtual void writeTo(ByteArray& bytes) = 0;

  /*!
   * \brief Reads the serialized information contained in \a bytes.
   *
   * The \a bytes array must have been created by a call to writeTo().
   */
  virtual void readFrom(Span<const Byte> bytes) = 0;

  //! Prints the properties and their values to the stream \a o
  virtual void print(std::ostream& o) const = 0;

  /*!
   * \brief Observable for writing.
   *
   * The observers registered in this observable are called
   * at the beginning of writeTo().
   */
  virtual IObservable* writeObservable() = 0;

  /*!
   * \brief Observable for reading.
   *
   * The observers registered in this observable are called
   * at the end of readFrom().
   */
  virtual IObservable* readObservable() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
