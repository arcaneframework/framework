// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TextReader.h                                                (C) 2000-2021 */
/*                                                                           */
/* Ecrivain de données.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_TEXTREADER_H
#define ARCANE_STD_TEXTREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IDeflateService;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe d'écriture d'un fichier texte pour les protections/reprises
 */
class TextReader
{
  class Impl;
 public:

  TextReader(const String& filename,bool is_binary);
  TextReader(const TextReader& rhs) = delete;
  ~TextReader();
  TextReader& operator=(const TextReader& rhs) = delete;

 public:

  void readIntegers(Span<Integer> values);

  void read(Span<Int16> values);
  void read(Span<Int32> values);
  void read(Span<Int64> values);
  void read(Span<Real> values);
  void read(Span<Byte> values);

 public:
  String fileName() const;
  void setFileOffset(Int64 v);
  void setDeflater(Ref<IDeflateService> ds);
 private:
  Impl* m_p;
 private:
  void _removeComments();
  Integer _getInteger();
  Int16 _getInt16();
  Int32 _getInt32();
  Int64 _getInt64();
  Real _getReal();
  void _binaryRead(void* bytes,Int64 len);
  void _checkStream(const char* type,Int64 nb_read_value);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe d'écriture d'un fichier texte pour les protections/reprises
 */
class KeyValueTextReader
{
  class Impl;

 public:

  KeyValueTextReader(const String& filename,bool is_binary,Int32 version);
  KeyValueTextReader(const KeyValueTextReader& rhs) = delete;
  ~KeyValueTextReader();
  KeyValueTextReader& operator=(const KeyValueTextReader& rhs) = delete;

 public:

  void readIntegers(const String& key,Span<Integer> values);

  void read(const String& key,Span<Int16> values);
  void read(const String& key,Span<Int32> values);
  void read(const String& key,Span<Int64> values);
  void read(const String& key,Span<Real> values);
  void read(const String& key,Span<Byte> values);

 public:
  String fileName() const;
  void setFileOffset(Int64 v);
  void setDeflater(Ref<IDeflateService> ds);
 private:
  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
