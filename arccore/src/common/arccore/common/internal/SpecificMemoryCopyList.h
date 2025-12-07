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
#include "arccore/common/CommonGlobal.h"

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
/*!
 * \brief Classe singleton contenant l'instance à utiliser pour les copies.
 *
 * Par défaut, l'instance est définie dans 'SpecificMemoryCopy.cc' et ne
 * gère que les copies vers/depuis un CPU.
 * Si un runtime accélérateur est initialisé, il peut remplacer l'instance
 * par défaut pour gérer les copies entre CPU et accélérateur.
 */
class ARCCORE_COMMON_EXPORT GlobalMemoryCopyList
{
 private:

  static ISpecificMemoryCopyList* default_global_copy_list;
  static ISpecificMemoryCopyList* accelerator_global_copy_list;

 public:

  //! Retourne l'instance par défaut pour la file \a queue
  static ISpecificMemoryCopyList* getDefault(const RunQueue* queue);

  /*!
   * \brief Positionne l'instance par défaut pour les copies
   * lorsqu'un runtime accélérateur est activé
   *
   * L'instance doit rester valide pendant toute la durée du programme.
   * 
   * Cette méthode est normalement appelée par l'API accélérateur pour
   * fournir des noyaux de copie spécifiques à chaque device.
   */
  static void setAcceleratorInstance(ISpecificMemoryCopyList* ptr);
  static ISpecificMemoryCopyList* acceleratorInstance()
  {
    return accelerator_global_copy_list;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
