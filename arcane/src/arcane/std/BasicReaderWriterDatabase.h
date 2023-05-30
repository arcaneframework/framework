// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicReaderWriterDatabase.h                                 (C) 2000-2021 */
/*                                                                           */
/* Base de donnée pour le service 'BasicReaderWriter'.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_BASICREADERWRITERDATABASE_H
#define ARCANE_STD_BASICREADERWRITERDATABASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IDataCompressor;
}

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * Utilisation d'un TextWriter avec écriture sous la forme (clé,valeur).
 *
 * Pour chaque valeur à écrire, il faut d'abord appeler setExtents() pour
 * positionner les dimensions de la donnée puis write() pour écrire les
 * valeurs. Cela est nécessaire pour conserver la compatibilité avec les
 * versions 1 et 2 du format où les données étaient écrites de manière
 * séquentielles.
 */
class KeyValueTextWriter
{
  class Impl;

 public:

  explicit KeyValueTextWriter(const String& filename, Int32 version);
  KeyValueTextWriter(const KeyValueTextWriter& rhs) = delete;
  ~KeyValueTextWriter();
  KeyValueTextWriter& operator=(const KeyValueTextWriter& rhs) = delete;

 public:

  void setExtents(const String& key_name, SmallSpan<const Int64> extents);
  void write(const String& key, Span<const std::byte> values);

 public:

  String fileName() const;
  void setDataCompressor(Ref<IDataCompressor> dc);
  Ref<IDataCompressor> dataCompressor() const;
  Int64 fileOffset();

 private:

  Impl* m_p;
  void _addKey(const String& key, SmallSpan<const Int64> extents);
  void _writeKey(const String& key);
  void _writeHeader();
  void _writeEpilog();
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

  KeyValueTextReader(const String& filename, Int32 version);
  KeyValueTextReader(const KeyValueTextReader& rhs) = delete;
  ~KeyValueTextReader();
  KeyValueTextReader& operator=(const KeyValueTextReader& rhs) = delete;

 public:

  void getExtents(const String& key_name, SmallSpan<Int64> extents);
  void readIntegers(const String& key, Span<Integer> values);

  void read(const String& key, Span<Int16> values);
  void read(const String& key, Span<Int32> values);
  void read(const String& key, Span<Int64> values);
  void read(const String& key, Span<Real> values);
  void read(const String& key, Span<Byte> values);

 public:

  String fileName() const;
  void setFileOffset(Int64 v);
  void setDataCompressor(Ref<IDataCompressor> ds);
  Ref<IDataCompressor> dataCompressor() const;

 private:

  Impl* m_p;

 private:

  void _readHeader();
  void _readJSON();
  void _readDirect(Int64 offset, Span<std::byte> bytes);
  void _setFileOffset(const String& key_name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
