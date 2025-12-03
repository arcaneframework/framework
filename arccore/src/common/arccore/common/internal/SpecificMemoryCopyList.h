// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SpecificMemoryCopyList.h                                    (C) 2000-2025 */
/*                                                                           */
/* Classe template pour gérer des fonctions spécialisées de copie mémoire.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_INTERNAL_SPECIFICMEMORYCOPYLIST_H
#define ARCCORE_COMMON_INTERNAL_SPECIFICMEMORYCOPYLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/base/ArrayExtentsValue.h"
#include "arccore/base/FatalErrorException.h"

#include <atomic>
#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arguments pour une copie de certains indices entre deux zones mémoire.
 */
class ARCCORE_COMMON_EXPORT IndexedMemoryCopyArgs
{
 public:

  using RunQueue = Arcane::Accelerator::RunQueue;

 public:

  IndexedMemoryCopyArgs(SmallSpan<const Int32> indexes, Span<const std::byte> source,
                        Span<std::byte> destination, const RunQueue* run_queue)
  : m_indexes(indexes)
  , m_source(source)
  , m_destination(destination)
  , m_queue(run_queue)
  {}

 public:

  SmallSpan<const Int32> m_indexes;
  Span<const std::byte> m_source;
  Span<std::byte> m_destination;
  const RunQueue* m_queue = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arguments pour une copie de certains indices vers/depuis
 * une zone mémoire multiple.
 */
class ARCCORE_COMMON_EXPORT IndexedMultiMemoryCopyArgs
{
 public:

  //! Constructeur pour copyTo
  IndexedMultiMemoryCopyArgs(SmallSpan<const Int32> indexes,
                             SmallSpan<const Span<const std::byte>> multi_memory,
                             Span<std::byte> destination,
                             RunQueue* run_queue)
  : m_indexes(indexes)
  , m_const_multi_memory(multi_memory)
  , m_destination_buffer(destination)
  , m_queue(run_queue)
  {}

  //! Constructor pour copyFrom
  IndexedMultiMemoryCopyArgs(SmallSpan<const Int32> indexes,
                             SmallSpan<Span<std::byte>> multi_memory,
                             Span<const std::byte> source,
                             RunQueue* run_queue)
  : m_indexes(indexes)
  , m_multi_memory(multi_memory)
  , m_source_buffer(source)
  , m_queue(run_queue)
  {}

 public:

  SmallSpan<const Int32> m_indexes;
  SmallSpan<const Span<const std::byte>> m_const_multi_memory;
  SmallSpan<Span<std::byte>> m_multi_memory;
  Span<const std::byte> m_source_buffer;
  Span<std::byte> m_destination_buffer;
  RunQueue* m_queue = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un copieur mémoire spécialisé pour une taille de donnée.
 */
class ARCCORE_COMMON_EXPORT ISpecificMemoryCopy
{
 public:

  virtual ~ISpecificMemoryCopy() = default;

 public:

  virtual void copyFrom(const IndexedMemoryCopyArgs& args) = 0;
  virtual void copyTo(const IndexedMemoryCopyArgs& args) = 0;
  virtual void fill(const IndexedMemoryCopyArgs& args) = 0;
  virtual void copyFrom(const IndexedMultiMemoryCopyArgs&) = 0;
  virtual void copyTo(const IndexedMultiMemoryCopyArgs&) = 0;
  virtual void fill(const IndexedMultiMemoryCopyArgs& args) = 0;
  virtual Int32 datatypeSize() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une liste d'instances de ISpecificMemoryCopy spécialisées.
 */
class ARCCORE_COMMON_EXPORT ISpecificMemoryCopyList
{
 public:

  /*!
   * \brief Positionne l'instance par défaut pour les copies.
   *
   * Cette méthode est normalement appelée par l'API accélérateur pour
   * fournir des noyaux de copie spécifiques à chaque device.
   */
  static void setDefaultCopyListIfNotSet(ISpecificMemoryCopyList* ptr);

 public:

  virtual void copyTo(Int32 datatype_size, const IndexedMemoryCopyArgs& args) = 0;
  virtual void copyFrom(Int32 datatype_size, const IndexedMemoryCopyArgs& args) = 0;
  virtual void fill(Int32 datatype_size, const IndexedMemoryCopyArgs& args) = 0;
  virtual void copyTo(Int32 datatype_size, const IndexedMultiMemoryCopyArgs& args) = 0;
  virtual void copyFrom(Int32 datatype_size, const IndexedMultiMemoryCopyArgs& args) = 0;
  virtual void fill(Int32 datatype_size, const IndexedMultiMemoryCopyArgs& args) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Liste d'instances de ISpecificMemoryCopy spécialisées.
 *
 * Cette classe contient des instances de ISpecificMemoryCopy spécialisées
 * pour une taille et un type de données. Cela permet au compilateur de
 * connaitre précisément la taille d'un type de donnée et ainsi de mieux
 * optimiser les boucles sans avoir besoin que toutes ces méthodes soient
 * templates et inline pour le développeur.
 */
template <typename Traits>
class SpecificMemoryCopyList
: public ISpecificMemoryCopyList
{
 public:

  using InterfaceType = typename Traits::InterfaceType;
  template <typename DataType, typename Extent> using SpecificType = typename Traits::template SpecificType<DataType, Extent>;
  using RefType = typename Traits::RefType;

 public:

  static constexpr Int32 NB_COPIER = 128;

 public:

  SpecificMemoryCopyList()
  {
    m_copier.fill(nullptr);

    addCopier<SpecificType<std::byte, ExtentValue<1>>>(); // 1
    addCopier<SpecificType<Int16, ExtentValue<1>>>(); // 2
    addCopier<SpecificType<std::byte, ExtentValue<3>>>(); // 3
    addCopier<SpecificType<Int32, ExtentValue<1>>>(); // 4
    addCopier<SpecificType<std::byte, ExtentValue<5>>>(); // 5
    addCopier<SpecificType<Int16, ExtentValue<3>>>(); // 6
    addCopier<SpecificType<std::byte, ExtentValue<7>>>(); // 7
    addCopier<SpecificType<Int64, ExtentValue<1>>>(); // 8
    addCopier<SpecificType<std::byte, ExtentValue<9>>>(); // 9
    addCopier<SpecificType<Int16, ExtentValue<5>>>(); // 10
    addCopier<SpecificType<Int32, ExtentValue<3>>>(); // 12

    addCopier<SpecificType<Int64, ExtentValue<2>>>(); // 16
    addCopier<SpecificType<Int64, ExtentValue<3>>>(); // 24
    addCopier<SpecificType<Int64, ExtentValue<4>>>(); // 32
    addCopier<SpecificType<Int64, ExtentValue<5>>>(); // 40
    addCopier<SpecificType<Int64, ExtentValue<6>>>(); // 48
    addCopier<SpecificType<Int64, ExtentValue<7>>>(); // 56
    addCopier<SpecificType<Int64, ExtentValue<8>>>(); // 64
    addCopier<SpecificType<Int64, ExtentValue<9>>>(); // 72
  }

  ~SpecificMemoryCopyList()
  {
    for (ISpecificMemoryCopy* copier : m_dynamic_copier_list)
      delete copier;
  }

 public:

  template <typename CopierType>
  void addCopier()
  {
    auto* copier = new CopierType();
    m_copier[copier->datatypeSize()] = copier;
    m_dynamic_copier_list.push_back(copier);
  }

 public:

  void printStats()
  {
    std::cout << "SpecificMemory::nb_specialized=" << m_nb_specialized
              << " nb_generic=" << m_nb_generic << "\n";
  }

  void checkValid()
  {
    // Vérifie que les taille sont correctes
    for (Int32 i = 0; i < NB_COPIER; ++i) {
      auto* x = m_copier[i];
      if (x && (x->datatypeSize() != i))
        ARCCORE_FATAL("Incoherent datatype size v={0} expected={1}", x->datatypeSize(), i);
    }
  }

 private:

  RefType _copier(Int32 v)
  {
    if (v < 0)
      ARCCORE_FATAL("Bad value {0} for datasize", v);

    InterfaceType* x = nullptr;
    if (v < NB_COPIER)
      x = m_copier[v];
    if (x) {
      if (x->datatypeSize() != v)
        ARCCORE_FATAL("Incoherent datatype size v={0} expected={1}", x->datatypeSize(), v);
      ++m_nb_specialized;
    }
    else
      ++m_nb_generic;
    return RefType(x, v);
  }

 public:

  void copyTo(Int32 datatype_size, const IndexedMemoryCopyArgs& args) override
  {
    auto c = _copier(datatype_size);
    c.copyTo(args);
  }
  void copyTo(Int32 datatype_size, const IndexedMultiMemoryCopyArgs& args) override
  {
    auto c = _copier(datatype_size);
    c.copyTo(args);
  }
  void copyFrom(Int32 datatype_size, const IndexedMemoryCopyArgs& args) override
  {
    auto c = _copier(datatype_size);
    c.copyFrom(args);
  }
  void copyFrom(Int32 datatype_size, const IndexedMultiMemoryCopyArgs& args) override
  {
    auto c = _copier(datatype_size);
    c.copyFrom(args);
  }
  void fill(Int32 datatype_size, const IndexedMemoryCopyArgs& args) override
  {
    auto c = _copier(datatype_size);
    c.fill(args);
  }
  void fill(Int32 datatype_size, const IndexedMultiMemoryCopyArgs& args) override
  {
    auto c = _copier(datatype_size);
    c.fill(args);
  }

 private:

  std::array<InterfaceType*, NB_COPIER> m_copier;
  std::atomic<Int32> m_nb_specialized = 0;
  std::atomic<Int32> m_nb_generic = 0;
  //! Liste des copieurs qu'il faudra supprimer via 'delete'
  std::vector<ISpecificMemoryCopy*> m_dynamic_copier_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType, typename Extent>
class SpecificMemoryCopyBase
: public ISpecificMemoryCopy
{
  static Int32 typeSize() { return static_cast<Int32>(sizeof(DataType)); }

 public:

  Int32 datatypeSize() const override { return m_extent.v * typeSize(); }

 public:

  Extent m_extent;

 protected:

  static Span<const DataType> _toTrueType(Span<const std::byte> a)
  {
    return { reinterpret_cast<const DataType*>(a.data()), a.size() / typeSize() };
  }
  static Span<DataType> _toTrueType(Span<std::byte> a)
  {
    return { reinterpret_cast<DataType*>(a.data()), a.size() / typeSize() };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType, typename Extent>
class SpecificMemoryCopy
: public SpecificMemoryCopyBase<DataType, Extent>
{
  using BaseClass = SpecificMemoryCopyBase<DataType, Extent>;
  using BaseClass::_toTrueType;

 public:

  using BaseClass::m_extent;

 public:

  void copyFrom(const IndexedMemoryCopyArgs& args) override
  {
    _copyFrom(args.m_indexes, _toTrueType(args.m_source), _toTrueType(args.m_destination));
  }
  void copyTo(const IndexedMemoryCopyArgs& args) override
  {
    _copyTo(args.m_indexes, _toTrueType(args.m_source), _toTrueType(args.m_destination));
  }
  void fill(const IndexedMemoryCopyArgs& args) override
  {
    _fill(args.m_indexes, _toTrueType(args.m_source), _toTrueType(args.m_destination));
  }
  void copyFrom(const IndexedMultiMemoryCopyArgs& args) override
  {
    _copyFrom(args.m_indexes, args.m_multi_memory, _toTrueType(args.m_source_buffer));
  }
  void copyTo(const IndexedMultiMemoryCopyArgs& args) override
  {
    _copyTo(args.m_indexes, args.m_const_multi_memory, _toTrueType(args.m_destination_buffer));
  }
  void fill(const IndexedMultiMemoryCopyArgs& args) override
  {
    _fill(args.m_indexes, args.m_multi_memory, _toTrueType(args.m_source_buffer));
  }

 public:

  void _copyFrom(SmallSpan<const Int32> indexes, Span<const DataType> source,
                 Span<DataType> destination)
  {
    ARCCORE_CHECK_POINTER(indexes.data());
    ARCCORE_CHECK_POINTER(source.data());
    ARCCORE_CHECK_POINTER(destination.data());

    Int32 nb_index = indexes.size();
    for (Int32 i = 0; i < nb_index; ++i) {
      Int64 z_index = (Int64)i * m_extent.v;
      Int64 zci = (Int64)(indexes[i]) * m_extent.v;
      for (Int32 z = 0, n = m_extent.v; z < n; ++z)
        destination[z_index + z] = source[zci + z];
    }
  }
  void _copyFrom(SmallSpan<const Int32> indexes, SmallSpan<Span<std::byte>> multi_views,
                 Span<const DataType> source)
  {
    ARCCORE_CHECK_POINTER(indexes.data());
    ARCCORE_CHECK_POINTER(source.data());
    ARCCORE_CHECK_POINTER(multi_views.data());

    const Int32 value_size = indexes.size() / 2;
    for (Int32 i = 0; i < value_size; ++i) {
      Int32 index0 = indexes[i * 2];
      Int32 index1 = indexes[(i * 2) + 1];
      Span<std::byte> orig_view_bytes = multi_views[index0];
      auto* orig_view_data = reinterpret_cast<DataType*>(orig_view_bytes.data());
      // Utilise un span pour tester les débordements de tableau mais on
      // pourrait directement utiliser 'orig_view_data' pour plus de performances
      Span<DataType> orig_view = { orig_view_data, orig_view_bytes.size() / (Int64)sizeof(DataType) };
      Int64 zci = ((Int64)(index1)) * m_extent.v;
      Int64 z_index = (Int64)i * m_extent.v;
      for (Int32 z = 0, n = m_extent.v; z < n; ++z)
        orig_view[zci + z] = source[z_index + z];
    }
  }

  /*!
   * \brief Remplit les valeurs d'indices spécifiés par \a indexes.
   *
   * Si \a indexes est vide, remplit toutes les valeurs.
   */
  void _fill(SmallSpan<const Int32> indexes, Span<const DataType> source,
             Span<DataType> destination)
  {
    ARCCORE_CHECK_POINTER(source.data());
    ARCCORE_CHECK_POINTER(destination.data());

    // Si \a indexes est vide, cela signifie qu'on copie toutes les valeurs
    Int32 nb_index = indexes.size();
    if (nb_index == 0) {
      Int64 nb_value = destination.size() / m_extent.v;
      for (Int64 i = 0; i < nb_value; ++i) {
        Int64 zci = i * m_extent.v;
        for (Int32 z = 0, n = m_extent.v; z < n; ++z)
          destination[zci + z] = source[z];
      }
    }
    else {
      ARCCORE_CHECK_POINTER(indexes.data());
      for (Int32 i = 0; i < nb_index; ++i) {
        Int64 zci = (Int64)(indexes[i]) * m_extent.v;
        for (Int32 z = 0, n = m_extent.v; z < n; ++z)
          destination[zci + z] = source[z];
      }
    }
  }

  void _fill(SmallSpan<const Int32> indexes, SmallSpan<Span<std::byte>> multi_views,
             Span<const DataType> source)
  {
    ARCCORE_CHECK_POINTER(source.data());
    ARCCORE_CHECK_POINTER(multi_views.data());

    const Int32 nb_index = indexes.size() / 2;
    if (nb_index == 0) {
      // Remplit toutes les valeurs du tableau avec la source.
      const Int32 nb_dim1 = multi_views.size();
      for (Int32 zz = 0; zz < nb_dim1; ++zz) {
        Span<std::byte> orig_view_bytes = multi_views[zz];
        Int64 nb_value = orig_view_bytes.size() / ((Int64)sizeof(DataType));
        auto* orig_view_data = reinterpret_cast<DataType*>(orig_view_bytes.data());
        Span<DataType> orig_view = { orig_view_data, nb_value };
        for (Int64 i = 0; i < nb_value; i += m_extent.v) {
          // Utilise un span pour tester les débordements de tableau mais on
          // pourrait directement utiliser 'orig_view_data' pour plus de performances
          for (Int32 z = 0, n = m_extent.v; z < n; ++z) {
            orig_view[i + z] = source[z];
          }
        }
      }
    }
    else {
      ARCCORE_CHECK_POINTER(indexes.data());
      for (Int32 i = 0; i < nb_index; ++i) {
        Int32 index0 = indexes[i * 2];
        Int32 index1 = indexes[(i * 2) + 1];
        Span<std::byte> orig_view_bytes = multi_views[index0];
        auto* orig_view_data = reinterpret_cast<DataType*>(orig_view_bytes.data());
        // Utilise un span pour tester les débordements de tableau mais on
        // pourrait directement utiliser 'orig_view_data' pour plus de performances
        Span<DataType> orig_view = { orig_view_data, orig_view_bytes.size() / (Int64)sizeof(DataType) };
        Int64 zci = ((Int64)(index1)) * m_extent.v;
        for (Int32 z = 0, n = m_extent.v; z < n; ++z)
          orig_view[zci + z] = source[z];
      }
    }
  }

  void _copyTo(SmallSpan<const Int32> indexes, Span<const DataType> source,
               Span<DataType> destination)
  {
    ARCCORE_CHECK_POINTER(indexes.data());
    ARCCORE_CHECK_POINTER(source.data());
    ARCCORE_CHECK_POINTER(destination.data());

    Int32 nb_index = indexes.size();

    for (Int32 i = 0; i < nb_index; ++i) {
      Int64 z_index = (Int64)i * m_extent.v;
      Int64 zci = (Int64)(indexes[i]) * m_extent.v;
      for (Int32 z = 0, n = m_extent.v; z < n; ++z)
        destination[zci + z] = source[z_index + z];
    }
  }

  void _copyTo(SmallSpan<const Int32> indexes, SmallSpan<const Span<const std::byte>> multi_views,
               Span<DataType> destination)
  {
    ARCCORE_CHECK_POINTER(indexes.data());
    ARCCORE_CHECK_POINTER(destination.data());
    ARCCORE_CHECK_POINTER(multi_views.data());

    const Int32 value_size = indexes.size() / 2;
    for (Int32 i = 0; i < value_size; ++i) {
      Int32 index0 = indexes[i * 2];
      Int32 index1 = indexes[(i * 2) + 1];
      Span<const std::byte> orig_view_bytes = multi_views[index0];
      auto* orig_view_data = reinterpret_cast<const DataType*>(orig_view_bytes.data());
      // Utilise un span pour tester les débordements de tableau mais on
      // pourrait directement utiliser 'orig_view_data' pour plus de performances
      Span<const DataType> orig_view = { orig_view_data, orig_view_bytes.size() / (Int64)sizeof(DataType) };
      Int64 zci = ((Int64)(index1)) * m_extent.v;
      Int64 z_index = (Int64)i * m_extent.v;
      for (Int32 z = 0, n = m_extent.v; z < n; ++z)
        destination[z_index + z] = orig_view[zci + z];
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Référence sur un copieur.
 *
 * Cette classe permet d'utiliser le copieur spécifique à une taille d'élément
 * s'il est disponible. Sinon on utilise un copieur générique.
 */
template <typename Traits>
class SpecificMemoryCopyRef
{
  template <typename DataType, typename Extent> using SpecificType = typename Traits::template SpecificType<DataType, Extent>;

 public:

  SpecificMemoryCopyRef(ISpecificMemoryCopy* specialized_copier, Int32 datatype_size)
  : m_specialized_copier(specialized_copier)
  , m_used_copier(specialized_copier)
  {
    m_generic_copier.m_extent.v = datatype_size;
    if (!m_used_copier)
      m_used_copier = &m_generic_copier;
  }

  void copyFrom(const IndexedMemoryCopyArgs& args)
  {
    m_used_copier->copyFrom(args);
  }

  void copyTo(const IndexedMemoryCopyArgs& args)
  {
    m_used_copier->copyTo(args);
  }

  void fill(const IndexedMemoryCopyArgs& args)
  {
    m_used_copier->fill(args);
  }

  void copyFrom(const IndexedMultiMemoryCopyArgs& args)
  {
    m_used_copier->copyFrom(args);
  }

  void copyTo(const IndexedMultiMemoryCopyArgs& args)
  {
    m_used_copier->copyTo(args);
  }

  void fill(const IndexedMultiMemoryCopyArgs& args)
  {
    m_used_copier->fill(args);
  }

 private:

  ISpecificMemoryCopy* m_specialized_copier = nullptr;
  SpecificType<std::byte, ExtentValue<DynExtent>> m_generic_copier;
  ISpecificMemoryCopy* m_used_copier = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
