// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ReferenceCounterImpl.h                                      (C) 2000-2025 */
/*                                                                           */
/* Implémentations liées au gestionnaire de compteur de référence.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_REFERENCECOUNTERIMPL_H
#define ARCCORE_BASE_REFERENCECOUNTERIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounter.h"
#include "arccore/base/RefBase.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file ReferenceCounterImpl.h
 *
 * Ce fichier ne doit être inclus que par les classes implémentations
 * utilisant un compte de référence. Pour les déclarations, il faut utiliser
 * le fichier 'RefDeclarations.h'
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T> ARCCORE_EXPORT void
ExternalReferenceCounterAccessor<T>::
addReference(T* t)
{
  if constexpr (::Arcane::impl::HasInternalAddReference<T>::value)
    t->_internalAddReference();
  else
    t->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T> ARCCORE_EXPORT void
ExternalReferenceCounterAccessor<T>::
removeReference(T* t)
{
  if constexpr (::Arcane::impl::HasInternalRemoveReference<T>::value) {
    bool need_destroy = t->_internalRemoveReference();
    if (need_destroy)
      delete t;
  }
  else
    t->removeReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation thread-safe d'un compteur de référence.
 *
 * L'implémentation utilise un std::atomic pour conserver le nombre de
 * références.
 *
 * La méthode removeReference() détruit l'instance lorsque ce compteur
 * de référence atteint 0.
 *
 * Cette classe est interne à Arcane.
 */
class ARCCORE_BASE_EXPORT ReferenceCounterImpl
{
  template <typename InstanceType> friend class impl::ReferenceCounterWrapper;

 public:

  virtual ~ReferenceCounterImpl() = default;

 public:

  // TODO: Rendre obsolète
  void addReference()
  {
    ++m_nb_ref;
  }

  // TODO: Rendre obsolète
  void removeReference()
  {
    // Décrémente et retourne la valeur d'avant.
    // Si elle vaut 1, cela signifie qu'on n'a plus de références
    // sur l'objet et qu'il faut le détruire.
    Int32 v = std::atomic_fetch_add(&m_nb_ref, -1);
    if (v == 1) {
      if (_destroyThisReference())
        delete this;
    }
  }

 public:

  void _internalAddReference()
  {
    ++m_nb_ref;
  }
  bool _internalRemoveReference()
  {
    // Décrémente et retourne la valeur d'avant.
    // Si elle vaut 1, cela signifie qu'on n'a plus de références
    // sur l'objet et qu'il faut éventuellement le détruire.
    Int32 v = std::atomic_fetch_add(&m_nb_ref, -1);
    if (v == 1)
      return _destroyThisReference();
    return false;
  }

 private:

  // Méthodes accessibles uniquement depuis ReferenceCounterWrapper
  void _setExternalDeleter(RefBase::DeleterBase* v)
  {
    delete m_external_deleter;
    m_external_deleter = v;
  }
  RefBase::DeleterBase* _externalDeleter() const
  {
    return m_external_deleter;
  }

 private:

  std::atomic<Int32> m_nb_ref = 0;
  RefBase::DeleterBase* m_external_deleter = nullptr;

 private:

  //! Retourne \a true si l'instance doit être détruite par l'appel à operator delete()
  bool _destroyThisReference()
  {
    if (!m_external_deleter)
      return true;
    bool do_delete = false;
    if (!m_external_deleter->m_no_destroy) {
      bool is_destroyed = m_external_deleter->_destroyHandle(this, m_external_deleter->m_handle);
      if (!is_destroyed) {
        do_delete = true;
      }
    }
    delete m_external_deleter;
    m_external_deleter = nullptr;
    return do_delete;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * Macro générique pour définir les méthodes gérant les compteurs de référence.
 */
#define ARCCORE_INTERNAL_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS(OPTIONAL_OVERRIDE) \
 private: \
\
  using BaseCounterType = ::Arcane::ReferenceCounterImpl; \
\
 public: \
\
  BaseCounterType* _internalReferenceCounter() OPTIONAL_OVERRIDE \
  { \
    return this; \
  } \
  void _internalAddReference() OPTIONAL_OVERRIDE \
  { \
    BaseCounterType::_internalAddReference(); \
  } \
  bool _internalRemoveReference() OPTIONAL_OVERRIDE \
  { \
    return BaseCounterType::_internalRemoveReference(); \
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro pour définir les méthodes gérant les compteurs
 * de référence.
 *
 * Cette macro s'utilise dans une classe implémentant une interface
 * pour laquelle la macro ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS()
 * a été utilisée. La classe implémentation doit dériver de
 * ReferenceCounterImpl. Par exemple:
 *
 * \code
 * class MyClass
 * : public ReferenceCounterImpl
 * , public IMyInterface
 * {
 *   ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();
 *  public:
 *   void myMethod1() override;
 * };
 * \endcode
 */
#define ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS() \
  ARCCORE_INTERNAL_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS(override)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro pour définir les méthodes et types une classe qui
 * utilise un compteur de référence.
 *
 * Cette macro doit être utilisée pour une classe pour laquelle on
 * a utilisé la macro ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(). Elle doit
 * se trouver dans une seule unité de translation (un fichier '.cc' par
 * exemple) et être utilisée dans le namespace Arccore. Par exemple:
 *
 * \code
 * namespace Arccore
 * {
 *   ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(MyNamespace::MyClass);
 * }
 * \endcode
 */
#define ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(class_name) \
  template class ExternalReferenceCounterAccessor<class_name>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
