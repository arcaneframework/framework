// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TextWriter.h                                                (C) 2000-2021 */
/*                                                                           */
/* Ecrivain de données.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_TEXTWRITER_H
#define ARCANE_STD_TEXTWRITER_H
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
class TextWriter
{
  class Impl;

 public:

  explicit TextWriter(const String& filename);
  TextWriter(const TextWriter& rhs) = delete;
  TextWriter();
  ~TextWriter();
  TextWriter& operator=(const TextWriter& rhs) = delete;

 public:

  void open(const String& filename);
  void write(Span<const std::byte> values);

 public:

  String fileName() const;
  void setDataCompressor(Ref<IDataCompressor> ds);
  Ref<IDataCompressor> dataCompressor() const;
  Int64 fileOffset();
  std::ostream& stream();

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Use write(Span<const std::byte>) instead")
  void write(Span<const Real> values);
  ARCANE_DEPRECATED_REASON("Y2023: Use write(Span<const std::byte>) instead")
  void write(Span<const Int16> values);
  ARCANE_DEPRECATED_REASON("Y2023: Use write(Span<const std::byte>) instead")
  void write(Span<const Int32> values);
  ARCANE_DEPRECATED_REASON("Y2023: Use write(Span<const std::byte>) instead")
  void write(Span<const Int64> values);
  ARCANE_DEPRECATED_REASON("Y2023: Use write(Span<const std::byte>) instead")
  void write(Span<const Byte> values);

 private:

  Impl* m_p;

 private:

  void _binaryWrite(const void* bytes, Int64 len);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
