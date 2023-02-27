// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryView.h                                                (C) 2000-2023 */
/*                                                                           */
/* Vues constantes ou modifiables sur une zone mémoire.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MEMORYVIEW_H
#define ARCANE_UTILS_MEMORYVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue constante sur une zone mémoire contigue.
 *
 * \warning API en cours de définition. Ne pas utiliser en dehors de Arcane.
 */
class ARCANE_UTILS_EXPORT MemoryView
{
 public:

  using SpanType = Span<const std::byte>;
  friend MutableMemoryView;

 public:

  MemoryView() = default;
  explicit constexpr MemoryView(Span<const std::byte> bytes)
  : m_bytes(bytes)
  , m_nb_element(bytes.size())
  , m_datatype_size(1)
  {}
  template <typename DataType> explicit constexpr MemoryView(Span<DataType> v)
  : MemoryView(v, 1)
  {}
  template <typename DataType> explicit constexpr MemoryView(ConstArrayView<DataType> v)
  : MemoryView(Span<const DataType>(v), 1)
  {}
  template <typename DataType> constexpr MemoryView(ConstArrayView<DataType> v, Int32 nb_component)
  : MemoryView(Span<const DataType>(v), nb_component)
  {}
  template <typename DataType> constexpr MemoryView(Span<DataType> v, Int32 nb_component)
  : m_bytes(asBytes(v))
  , m_nb_element(v.size())
  , m_datatype_size(static_cast<Int32>(sizeof(DataType)) * nb_component)
  {}

 public:

  template <typename DataType> constexpr MemoryView&
  operator=(Span<DataType> v)
  {
    m_bytes = asBytes(v);
    m_nb_element = v.size();
    m_datatype_size = static_cast<Int32>(sizeof(DataType));
    return (*this);
  }

 private:

  constexpr MemoryView(Span<const std::byte> bytes, Int32 datatype_size, Int64 nb_element)
  : m_bytes(bytes)
  , m_nb_element(nb_element)
  , m_datatype_size(datatype_size)
  {}

 public:

  //! Vue sous forme d'octets
  constexpr SpanType bytes() const { return m_bytes; }

  //! Pointeur sur la zone mémoire
  constexpr const std::byte* data() const { return m_bytes.data(); }

  //! Nombre d'éléments
  constexpr Int64 nbElement() const { return m_nb_element; }

  //! Taille du type de donnée associé (1 par défaut)
  constexpr Int32 datatypeSize() const { return m_datatype_size; }

  //! Sous-vue à partir de l'indice \a begin_index et contenant \a nb_element
  constexpr MemoryView subView(Int64 begin_index, Int64 nb_element) const
  {
    Int64 byte_offset = begin_index * m_datatype_size;
    auto sub_bytes = m_bytes.subspan(byte_offset, nb_element * m_datatype_size);
    return MemoryView(sub_bytes, m_datatype_size, nb_element);
  }

 public:

  /*!
   * \brief Copie dans l'instance indexée les données de \a v.
   *
   * L'opération est équivalente au pseudo-code suivant:
   *
   * \code
   * Int64 n = indexes.size();
   * for( Int64 i=0; i<n; ++i )
   *   v[indexes[i]] = this[i];
   * \endcode
   *
   * \pre this.datatypeSize() == v.datatypeSize();
   * \pre v.nbElement() >= indexes.size();
   */
  void copyToIndexesHost(MutableMemoryView v, Span<const Int32> indexes);

 public:

  //! Vue convertie en un Span
  ARCANE_DEPRECATED_REASON("Use bytes() instead")
  SpanType span() const { return m_bytes; }

  ARCANE_DEPRECATED_REASON("Use bytes().size() instead")
  constexpr Int64 size() const { return m_bytes.size(); }

 private:

  SpanType m_bytes;
  Int64 m_nb_element = 0;
  Int32 m_datatype_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue modifiable sur une zone mémoire contigue.
 * \ingroup MemoryView
 */
class ARCANE_UTILS_EXPORT MutableMemoryView
{
 public:

  using SpanType = Span<std::byte>;

 public:

  MutableMemoryView() = default;
  explicit constexpr MutableMemoryView(SpanType bytes)
  : m_bytes(bytes)
  , m_nb_element(bytes.size())
  , m_datatype_size(1)
  {}
  template <typename DataType> explicit constexpr MutableMemoryView(Span<DataType> v)
  : MutableMemoryView(v, 1)
  {}
  template <typename DataType> explicit constexpr MutableMemoryView(ArrayView<DataType> v)
  : MutableMemoryView(Span<DataType>(v), 1)
  {}
  template <typename DataType> explicit constexpr MutableMemoryView(ArrayView<DataType> v, Int32 nb_component)
  : MutableMemoryView(Span<DataType>(v), nb_component)
  {}
  template <typename DataType> constexpr MutableMemoryView(Span<DataType> v, Int32 nb_component)
  : m_bytes(asWritableBytes(v))
  , m_nb_element(v.size())
  , m_datatype_size(static_cast<Int32>(sizeof(DataType)) * nb_component)
  {}

 public:

  template <typename DataType> constexpr MutableMemoryView&
  operator=(Span<DataType> v)
  {
    m_bytes = asWritableBytes(v);
    m_nb_element = v.size();
    m_datatype_size = static_cast<Int32>(sizeof(DataType));
    return (*this);
  }

 private:

  constexpr MutableMemoryView(Span<std::byte> bytes, Int32 datatype_size, Int64 nb_element)
  : m_bytes(bytes)
  , m_nb_element(nb_element)
  , m_datatype_size(datatype_size)
  {}

 public:

  constexpr operator MemoryView() const { return MemoryView(m_bytes, m_datatype_size, m_nb_element); }

 public:

  //! Vue sous forme d'octets
  constexpr SpanType bytes() const { return m_bytes; }

  //! Pointeur sur la zone mémoire
  constexpr std::byte* data() const { return m_bytes.data(); }

  //! Nombre d'éléments
  constexpr Int64 nbElement() const { return m_nb_element; }

  //! Taille du type de donnée associé (1 par défaut)
  constexpr Int32 datatypeSize() const { return m_datatype_size; }

  //! Sous-vue à partir de l'indice \a begin_index
  constexpr MutableMemoryView subView(Int64 begin_index, Int64 nb_element) const
  {
    Int64 byte_offset = begin_index * m_datatype_size;
    auto sub_bytes = m_bytes.subspan(byte_offset, nb_element * m_datatype_size);
    return MutableMemoryView(sub_bytes, m_datatype_size, nb_element);
  }

 public:

  /*!
   * \brief Copie dans l'instance les données de \a v.
   *
   * Utilise std::memmove pour la copie.
   *
   * \pre v.bytes.size() >= bytes.size()
   */
  void copyHost(MemoryView v);

  /*!
   * \brief Copie dans l'instance les données indexées de \a v.
   *
   * L'opération est équivalente au pseudo-code suivant:
   *
   * \code
   * Int64 n = indexes.size();
   * for( Int64 i=0; i<n; ++i )
   *   this[i] = v[indexes[i]];
   * \endcode
   *
   * \pre this.datatypeSize() == v.datatypeSize();
   * \pre this.nbElement() >= indexes.size();
   */
  void copyFromIndexesHost(MemoryView v, Span<const Int32> indexes);

 public:

  ARCANE_DEPRECATED_REASON("Use bytes() instead")
  constexpr SpanType span() const { return m_bytes; }

  ARCANE_DEPRECATED_REASON("Use bytes().size() instead")
  constexpr Int64 size() const { return m_bytes.size(); }

 private:

  SpanType m_bytes;
  Int64 m_nb_element = 0;
  Int32 m_datatype_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé une vue mémoire constante à partir d'un \a Span
template <typename DataType> MemoryView
makeMemoryView(Span<DataType> v)
{
  return MemoryView(v);
}

//! Créé une vue mémoire constante sur l'adresse \a v
template <typename DataType> MemoryView
makeMemoryView(const DataType* v)
{
  return MemoryView(Span<const DataType>(v, 1));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé une vue mémoire modifiable à partir d'un \a Span
template <typename DataType> MutableMemoryView
makeMutableMemoryView(Span<DataType> v)
{
  return MutableMemoryView(v);
}

//! Créé une vue mémoire modifiable sur l'adresse \a v
template <typename DataType> MutableMemoryView
makeMutableMemoryView(DataType* v)
{
  return MutableMemoryView(Span<DataType>(v, 1));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
