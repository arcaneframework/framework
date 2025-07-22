// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryView.h                                                (C) 2000-2025 */
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
 * \ingroup MemoryView
 * \brief Vue constante sur une zone mémoire contigue contenant des
 * éléments de taille fixe.
 *
 * Les fonctions makeConstMemoryView() permettent de créer des instances
 * de cette classe.
 *
 * \warning API en cours de définition. Ne pas utiliser en dehors de Arcane.
 */
class ARCANE_UTILS_EXPORT ConstMemoryView
{
  friend ARCANE_UTILS_EXPORT ConstMemoryView
  makeConstMemoryView(const void* ptr, Int32 datatype_size, Int64 nb_element);

 public:

  using SpanType = Span<const std::byte>;
  friend MutableMemoryView;

 public:

  ConstMemoryView() = default;
  explicit constexpr ConstMemoryView(Span<const std::byte> bytes)
  : m_bytes(bytes)
  , m_nb_element(bytes.size())
  , m_datatype_size(1)
  {}
  template <typename DataType> explicit constexpr ConstMemoryView(Span<DataType> v)
  : ConstMemoryView(Span<const DataType>(v), 1)
  {}
  template <typename DataType> explicit constexpr ConstMemoryView(Span<const DataType> v)
  : ConstMemoryView(v, 1)
  {}
  template <typename DataType> explicit constexpr ConstMemoryView(ConstArrayView<DataType> v)
  : ConstMemoryView(Span<const DataType>(v), 1)
  {}
  template <typename DataType> explicit constexpr ConstMemoryView(ArrayView<DataType> v)
  : ConstMemoryView(Span<const DataType>(v), 1)
  {}
  template <typename DataType> constexpr ConstMemoryView(ConstArrayView<DataType> v, Int32 nb_component)
  : ConstMemoryView(Span<const DataType>(v), nb_component)
  {}
  template <typename DataType> constexpr ConstMemoryView(ArrayView<DataType> v, Int32 nb_component)
  : ConstMemoryView(Span<const DataType>(v), nb_component)
  {}
  template <typename DataType> constexpr ConstMemoryView(Span<DataType> v, Int32 nb_component)
  : ConstMemoryView(Span<const DataType>(v), nb_component)
  {
  }
  template <typename DataType> constexpr ConstMemoryView(Span<const DataType> v, Int32 nb_component)
  : m_nb_element(v.size())
  , m_datatype_size(static_cast<Int32>(sizeof(DataType)) * nb_component)
  {
    auto x = asBytes(v);
    m_bytes = SpanType(x.data(), x.size() * nb_component);
  }

 public:

  template <typename DataType> constexpr ConstMemoryView&
  operator=(Span<DataType> v)
  {
    m_bytes = asBytes(v);
    m_nb_element = v.size();
    m_datatype_size = static_cast<Int32>(sizeof(DataType));
    return (*this);
  }

 private:

  constexpr ConstMemoryView(Span<const std::byte> bytes, Int32 datatype_size, Int64 nb_element)
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
  constexpr ConstMemoryView subView(Int64 begin_index, Int64 nb_element) const
  {
    Int64 byte_offset = begin_index * m_datatype_size;
    auto sub_bytes = m_bytes.subspan(byte_offset, nb_element * m_datatype_size);
    return { sub_bytes, m_datatype_size, nb_element };
  }

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
 * \ingroup MemoryView
 *
 * \brief Vue modifiable sur une zone mémoire contigue contenant des
 * éléments de taille fixe.
 *
 * Les fonctions makeMutableMemoryView() permettent de créer des instances
 * de cette classe.
 *
 * \warning API en cours de définition. Ne pas utiliser en dehors de Arcane.
 */
class ARCANE_UTILS_EXPORT MutableMemoryView
{
  friend ARCANE_UTILS_EXPORT MutableMemoryView
  makeMutableMemoryView(void* ptr, Int32 datatype_size, Int64 nb_element);

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
  : m_nb_element(v.size())
  , m_datatype_size(static_cast<Int32>(sizeof(DataType)) * nb_component)
  {
    auto x = asWritableBytes(v);
    m_bytes = SpanType(x.data(), x.size() * nb_component);
  }

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

  constexpr operator ConstMemoryView() const { return { m_bytes, m_datatype_size, m_nb_element }; }

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
    return { sub_bytes, m_datatype_size, nb_element };
  }

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
/*!
 * \ingroup MemoryView
 *
 * \brief Liste de vues constantes sur des zones mémoires contigues.
 *
 * \warning API en cours de définition. Ne pas utiliser en dehors de Arcane.
 */
class ARCANE_UTILS_EXPORT ConstMultiMemoryView
{
 public:

  ConstMultiMemoryView(SmallSpan<const Span<const std::byte>> views, Int32 datatype_size)
  : m_views(views)
  , m_datatype_size(datatype_size)
  {}
  ConstMultiMemoryView(SmallSpan<const Span<std::byte>> views, Int32 datatype_size)
  : m_datatype_size(datatype_size)
  {
    auto* ptr = reinterpret_cast<const Span<const std::byte>*>(views.data());
    m_views = { ptr, views.size() };
  }

 public:

  /*!
   * \brief Copie dans l'instance indexée les données de \a v.
   *
   * L'opération est équivalente au pseudo-code suivant:
   *
   * \code
   * Int32 n = indexes.size();
   * for( Int32 i=0; i<n; ++i ){
   *   Int32 index0 = indexes[ (i*2)   ];
   *   Int32 index1 = indexes[ (i*2)+1 ];
   *   v[i] = this[index0][index1];
   * }
   * \endcode
   *
   * Le tableau des indexes doit avoir une taille multiple de 2. Les valeurs
   * paires servent à indexer le premier tableau et les valeurs impaires le 2ème.
   *
   * Si \a run_queue n'est pas nul, elle sera utilisée pour la copie.
   *
   * \pre this.datatypeSize() == v.datatypeSize();
   * \pre v.nbElement() >= indexes.size();
   */
  void copyToIndexes(MutableMemoryView v, SmallSpan<const Int32> indexes,
                     RunQueue* run_queue = nullptr);

 private:

  SmallSpan<const Span<const std::byte>> m_views;
  Int32 m_datatype_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup MemoryView
 *
 * \brief Liste de vues modifiables sur des zones mémoires contigues.
 *
 * \warning API en cours de définition. Ne pas utiliser en dehors de Arcane.
 */
class ARCANE_UTILS_EXPORT MutableMultiMemoryView
{
 public:

  MutableMultiMemoryView(SmallSpan<Span<std::byte>> views, Int32 datatype_size)
  : m_views(views)
  , m_datatype_size(datatype_size)
  {}

 public:

  /*!
   * \brief Copie dans l'instance indexée les données de \a v.
   *
   * L'opération est équivalente au pseudo-code suivant:
   *
   * \code
   * Int32 n = indexes.size();
   * for( Int32 i=0; i<n; ++i ){
   *   Int32 index0 = indexes[ (i*2)   ];
   *   Int32 index1 = indexes[ (i*2)+1 ];
   *   this[index0][index1] = v[i];
   * }
   * \endcode
   *
   * Le tableau des indexes doit avoir une taille multiple de 2. Les valeurs
   * paires servent à indexer le premier tableau et les valeurs impaires le 2ème.
   *
   * Si \a run_queue n'est pas nul, elle sera utilisée pour la copie.
   *
   * \pre this.datatypeSize() == v.datatypeSize();
   * \pre v.nbElement() >= indexes.size();
   */
  void copyFromIndexes(ConstMemoryView v, SmallSpan<const Int32> indexes,
                       RunQueue* run_queue = nullptr);

  /*!
   * \brief Remplit dans l'instance les données de \a v.
   *
   * \a v doit avoir une seule valeur. Cette valeur sera utilisée
   * pour remplir les valeur de l'instance aux indices spécifiés par
   * \a indexes. Elle doit être accessible depuis l'hôte.
   *
   * L'opération est équivalente au pseudo-code suivant:
   *
   * \code
   * Int32 n = indexes.size();
   * for( Int32 i=0; i<n; ++i ){
   *   Int32 index0 = indexes[ (i*2)   ];
   *   Int32 index1 = indexes[ (i*2)+1 ];
   *   this[index0][index1] = v[0];
   * }
   *
   * Si \a run_queue n'est pas nul, elle sera utilisée pour la copie.
   *
   * \pre this.datatypeSize() == v.datatypeSize();
   * \pre this.nbElement() >= indexes.size();
   */
  void fillIndexes(ConstMemoryView v, SmallSpan<const Int32> indexes,
                   RunQueue* run_queue = nullptr);

  /*!
   * \brief Remplit les éléments de l'instance avec la valeur \a v.
   *
   * \a v doit avoir une seule valeur. Elle doit être accessible depuis l'hôte.
   *
   * L'opération est équivalente au pseudo-code suivant:
   *
   * \code
   * Int32 n = nbElement();
   * for( Int32 i=0; i<n; ++i ){
   *   Int32 index0 = (i*2);
   *   Int32 index1 = (i*2)+1;
   *   this[index0][index1] = v[0];
   * }
   *
   * Si \a run_queue n'est pas nul, elle sera utilisée pour la copie.
   *
   * \pre this.datatypeSize() == v.datatypeSize();
   */
  void fill(ConstMemoryView v, RunQueue* run_queue = nullptr);

 private:

  SmallSpan<Span<std::byte>> m_views;
  Int32 m_datatype_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé une vue mémoire constante à partir d'un \a Span
template <typename DataType> ConstMemoryView
makeMemoryView(Span<DataType> v)
{
  return ConstMemoryView(v);
}

//! Créé une vue mémoire constante sur l'adresse \a v
template <typename DataType> ConstMemoryView
makeMemoryView(const DataType* v)
{
  return ConstMemoryView(Span<const DataType>(v, 1));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé une vue mémoire modifiable à partir d'un \a Span
template <typename DataType> MutableMemoryView
makeMutableMemoryView(Span<DataType> v)
{
  return MutableMemoryView(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé une vue mémoire modifiable sur l'adresse \a v
template <typename DataType> MutableMemoryView
makeMutableMemoryView(DataType* v)
{
  return MutableMemoryView(Span<DataType>(v, 1));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une vue mémoire modifiable.
 *
 * \param ptr adresse de la zone mémoire.
 * \param datatype_size taille (en octet) du type de la donnée.
 * \param nb_element nombre d'éléments de la vue.
 *
 * La zone mémoire aura pour taille datatype_size * nb_element octets.
 */
extern "C++" ARCANE_UTILS_EXPORT MutableMemoryView
makeMutableMemoryView(void* ptr, Int32 datatype_size, Int64 nb_element);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une vue mémoire en lecture seule.
 *
 * \param ptr adresse de la zone mémoire.
 * \param datatype_size taille (en octet) du type de la donnée.
 * \param nb_element nombre d'éléments de la vue.
 *
 * La zone mémoire aura pour taille datatype_size * nb_element octets.
 */
extern "C++" ARCANE_UTILS_EXPORT ConstMemoryView
makeConstMemoryView(const void* ptr, Int32 datatype_size, Int64 nb_element);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
