// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ThreadPrivate.h                                             (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de conserver une valeur spécifique par thread.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_THREADPRIVATE_H
#define ARCCORE_CONCURRENCY_THREADPRIVATE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include "arccore/concurrency/GlibAdapter.h"

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conteneur pour les valeurs privées par thread.
 *
 * Il faut appeler initialize() avant d'utiliser les méthodes setValue()/getValue().
 * Cette méthode initialize() peut être appelée plusieurs fois.
 *
 * \deprecated Utiliser 'thread_local' du C++11.
 */
class ARCCORE_CONCURRENCY_EXPORT ThreadPrivateStorage
{
 public:

  ARCCORE_DEPRECATED_REASON("Y2022; This class is deprecated. Use 'thread_local' specifier.")
  ThreadPrivateStorage();
  ~ThreadPrivateStorage();

 public:

  /*!
   * \brief Initialise la clé contenant les valeurs par thread.
   * Cette méthode peut être appelée plusieurs fois et ne fait rien si
   * la clé a déjà été initialisée.
   *
   * \warning Cette méthode n'est pas thread-safe. L'utilisateur doit donc
   * faire attention lors du premier appel.
   */
  void initialize();
  void* getValue();
  void setValue(void* v);

 private:

  GlibPrivate* m_storage;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base permettant de conserveur une instance d'un objet par thread.
 *
 * \deprecated Utiliser 'thread_local' du C++11.
 */
class ARCCORE_CONCURRENCY_EXPORT ThreadPrivateBase
{
 public:

  class ICreateFunctor
  {
   public:

    virtual ~ICreateFunctor() {}
    virtual void* createInstance() = 0;
  };

 public:

  ARCCORE_DEPRECATED_REASON("Y2022; This class is deprecated. Use 'thread_local' specifier.")
  ThreadPrivateBase(ThreadPrivateStorage* key, ICreateFunctor* create_functor)
  : m_key(key)
  , m_create_functor(create_functor)
  {
  }

  ~ThreadPrivateBase()
  {
  }

 public:

  /*!
   * \brief Récupère l'instance spécifique au thread courant.
   *
   * Si cette dernière n'existe pas encore, elle est créé via
   * le functor passé en argument du constructeur.
   *
   * \warning Cette méthode ne doit pas être appelée tant que
   * la clé associée (ThreadPrivateStorage) n'a pas été initialisée
   * par l'apple à ThreadPrivateStorage::initialize().
   */
  void* item();

 private:

  ThreadPrivateStorage* m_key;
  GlibMutex m_mutex;
  ICreateFunctor* m_create_functor;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe permettant une instance d'un type par thread.
 *
 * Il faut passer en argument du constructeur le conteneur permettant
 * de conserver les valeurs. Ce conteneur doit avoir été initialisé
 * via ThreadPrivateStorage::initialize() avant d'utiliser cette classe.
 *
 * Cette classe ne possède qu'une seule méthode item()
 * permettant de récupérer une instance d'un type \a T par
 * thread. Au premier appel de item() pour un thread donné,
 * une instance de \a T est construite.
 * Le type \a T doit avoir un constructeur par défaut
 * et doit avoir une méthode \a build().
 * \threadsafeclass
 *
 * \deprecated Utiliser 'thread_local' du C++11.
 */
template <typename T>
class ThreadPrivate
: private ThreadPrivateBase::ICreateFunctor
{
 public:

  ARCCORE_DEPRECATED_REASON("Y2022; This class is deprecated. Use 'thread_local' specifier.")
  ThreadPrivate(ThreadPrivateStorage* key)
  : m_storage(key, this)
  {
  }

  ~ThreadPrivate()
  {
    for (T* item : m_allocated_items)
      delete item;
  }

 public:

  //! Instance spécifique au thread courant.
  T* item()
  {
    return (T*)(m_storage.item());
  }

 private:

  void* createInstance() override
  {
    T* new_ptr = new T();
    new_ptr->build();
    m_allocated_items.push_back(new_ptr);
    return new_ptr;
  }

 private:

  std::vector<T*> m_allocated_items;
  ThreadPrivateBase m_storage;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
