// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TextReader2.h                                                (C) 2000-2024 */
/*                                                                           */
/* Ecrivain de données.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_TEXTREADER2_H
#define ARCANE_STD_INTERNAL_TEXTREADER2_H
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
 * \brief Classe d'écriture d'un fichier texte pour les protections/reprises
 */
class TextReader2
{
  class Impl;

 public:

  explicit TextReader2(const String& filename);
  TextReader2(const TextReader2& rhs) = delete;
  ~TextReader2();
  TextReader2& operator=(const TextReader2& rhs) = delete;

 public:

  void read(Span<std::byte> values);
  void readIntegers(Span<Integer> values);

 public:

  String fileName() const;
  void setFileOffset(Int64 v);
  void setDataCompressor(Ref<IDataCompressor> ds);
  Ref<IDataCompressor> dataCompressor() const;
  std::istream& stream();
  Int64 fileLength() const;

 private:

  Impl* m_p;

 private:

  void _binaryRead(void* bytes, Int64 len);
  void _checkStream(const char* type, Int64 nb_read_value);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
