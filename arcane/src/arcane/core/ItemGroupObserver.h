// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupObserver.h                                         (C) 2000-2025 */
/*                                                                           */
/* Interface et implémentation basique des observeurs de groupe.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMGROUPOBSERVER_H
#define ARCANE_CORE_ITEMGROUPOBSERVER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IItemGroupObserver
{
 public:

  template <typename T>
  struct FuncTraits
  {
    //! Type du pointeur sur la méthode avec Infos
    typedef void (T::*FuncPtrWithInfo)(const Int32ConstArrayView* info);

    //! Type du pointeur sur la méthode sans Infos
    typedef void (T::*FuncPtr)();
  };

 public:

  //! Destructeur
  virtual ~IItemGroupObserver() = default;

  /*!
   * \brief Execute l'action associée à l'extension.
   *
   * \param info liste des localIds ajoutés
   * Suppose qu'il n'y a pas de changement d'ordre ou de renumérotation.
   *
   * Cette méthode ne peut pas être parallèle.
   */
  virtual void executeExtend(const Int32ConstArrayView* info) = 0;

  /*!
   * \brief Execute l'action associée à l'extension.
   *
   * \param info liste des positions supprimées dans l'ancien groupe
   * Suppose qu'il n'y a pas de changement d'ordre ou de renumérotation
   * Cette approche par rapport à la liste des localIds est motivée par
   * la contrainte dans PartialVariable qui n'a pas connaissance des localIds
   * qu'il héberge.
   * \param info2 liste des localIds des éléments supprimés. Potentiellement redondant
   * avec \a info, mais inévitable pour certaines structures changeant l'ordre par rapport
   * au groupe de référence (ex: ItemGroupDynamicMeshObserver) (DEPRECATED)
   *
   * Cette méthode ne peut pas être parallèle.
   */
  virtual void executeReduce(const Int32ConstArrayView* info) = 0;

  /*!
   * \brief Éxecute l'action associée au compactage.
   *
   * \param info liste des permutations dans le sens old->new
   * Suppose qu'il n'y a pas de changement de taille.
   */
  virtual void executeCompact(const Int32ConstArrayView* info) = 0;

  /*!
   * \brief Execute l'action associée à l'invalidation.
   *
   * Aucune information de transition disponible.
   */
  virtual void executeInvalidate() = 0;

  /*!
   * \brief Indique si l'observer aura besoin d'information de transition
   *
   * Cette information ne doit pas changer après le premier appel à cet fonction
   */
  virtual bool needInfo() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
class ItemGroupObserverWithInfoT
: public IItemGroupObserver
{
 public:

  ItemGroupObserverWithInfoT(T* object,
                             typename FuncTraits<T>::FuncPtrWithInfo extend_funcptr,
                             typename FuncTraits<T>::FuncPtrWithInfo reduce_funcptr,
                             typename FuncTraits<T>::FuncPtrWithInfo compact_funcptr,
                             typename FuncTraits<T>::FuncPtr invalidate_funcptr)
  : m_object(object)
  , m_extend_function(extend_funcptr)
  , m_reduce_function(reduce_funcptr)
  , m_compact_function(compact_funcptr)
  , m_invalidate_function(invalidate_funcptr)
  {}

 public:

  void executeExtend(const Int32ConstArrayView* info) override
  {
    (m_object->*m_extend_function)(info);
  }

  void executeReduce(const Int32ConstArrayView* info) override
  {
    (m_object->*m_reduce_function)(info);
  }

  void executeCompact(const Int32ConstArrayView* info) override
  {
    (m_object->*m_compact_function)(info);
  }

  void executeInvalidate() override
  {
    (m_object->*m_invalidate_function)();
  }

  bool needInfo() const override { return true; }

 private:

  T* m_object = nullptr; //!< Objet associé.
  typename FuncTraits<T>::FuncPtrWithInfo m_extend_function = nullptr; //!< Pointeur vers la méthode associée.
  typename FuncTraits<T>::FuncPtrWithInfo m_reduce_function = nullptr; //!< Pointeur vers la méthode associée.
  typename FuncTraits<T>::FuncPtrWithInfo m_compact_function = nullptr; //!< Pointeur vers la méthode associée.
  typename FuncTraits<T>::FuncPtr m_invalidate_function = nullptr; //!< Pointeur vers la méthode associée.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
class ItemGroupObserverWithoutInfoT
: public IItemGroupObserver
{
 public:

  //! Constructeur à partir d'une unique fonction sans argument
  ItemGroupObserverWithoutInfoT(T* object, typename FuncTraits<T>::FuncPtr funcptr)
  : m_object(object)
  , m_function(funcptr)
  {}

 public:

  void executeExtend(const Int32ConstArrayView*) override
  {
    (m_object->*m_function)();
  }

  void executeReduce(const Int32ConstArrayView*) override
  {
    (m_object->*m_function)();
  }

  void executeCompact(const Int32ConstArrayView*) override
  {
    (m_object->*m_function)();
  }

  void executeInvalidate() override
  {
    (m_object->*m_function)();
  }

  bool needInfo() const override
  {
    return false;
  }

 private:

  T* m_object = nullptr; //!< Objet associé.
  typename FuncTraits<T>::FuncPtr m_function = nullptr; //!< Pointeur vers la méthode associée.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Utilitaire pour création simplifié de ItemGroupObserverT
template <typename T>
inline IItemGroupObserver* newItemGroupObserverT(T* object,
                                                 typename IItemGroupObserver::FuncTraits<T>::FuncPtr funcptr)
{
  return new ItemGroupObserverWithoutInfoT<T>(object, funcptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Utilitaire pour création simplifié de ItemGroupObserverT
template <typename T> inline IItemGroupObserver*
newItemGroupObserverT(T* object,
                      typename IItemGroupObserver::FuncTraits<T>::FuncPtrWithInfo extend_funcptr,
                      typename IItemGroupObserver::FuncTraits<T>::FuncPtrWithInfo reduce_funcptr,
                      typename IItemGroupObserver::FuncTraits<T>::FuncPtrWithInfo compact_funcptr,
                      typename IItemGroupObserver::FuncTraits<T>::FuncPtr invalidate_funcptr)
{
  return new ItemGroupObserverWithInfoT<T>(object, extend_funcptr, reduce_funcptr, compact_funcptr, invalidate_funcptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
