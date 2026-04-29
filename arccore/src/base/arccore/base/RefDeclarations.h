// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RefDeclarations.h                                           (C) 2000-2026 */
/*                                                                           */
/* Déclarations liées à la gestion des références sur une instance.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_REFDECLARATIONS_H
#define ARCCORE_BASE_REFDECLARATIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

#include <type_traits>
#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file RefDeclarations.h
 *
 * Ce fichier contient les déclarations et macros pour gérer classes
 * utilisant les compteurs de référence. Pour l'implémentation il faut utiliser
 * le fichier 'ReferenceCounterImpl.h'
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// La classe ExternalReferenceCounterAccessor doit rester dans le namespace
// Arccore pour compatiblité avec l'existant et la macro
// ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS.
namespace Arccore
{
template <class T>
class ExternalReferenceCounterAccessor
{
 public:

  static ARCCORE_EXPORT void addReference(T* t);
  static ARCCORE_EXPORT void removeReference(T* t);
};
} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using Arccore::ExternalReferenceCounterAccessor;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Structure servant à tagger les interfaces/classes qui utilisent
 * un compteur de référence interne.
 *
 * Ce tag s'utilise via un typedef comme suit:
 *
 * \code
 * class MyClass
 * {
 *   public:
 *    using ReferenceCounterTagType = ReferenceCounterTag;
 *   public:
 *    void addReference();
 *    void removeReference();
 * };
 * \endcode
 */
struct ReferenceCounterTag
{};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

constexpr int REF_TAG_SHARED_PTR = 0;
constexpr int REF_TAG_REFERENCE_COUNTER = 1;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonction pour savoir quel type de compteur de référence
 * utilise une classe.
 *
 * Par défaut on utilise std::shared_ptr.
 * Pour utiliser un compteur de référence interne, il faur surcharger cette
 * méthode via la macro ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS().
 */
inline constexpr int arcaneImplGetRefTagId(void*)
{
  return REF_TAG_SHARED_PTR;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Caractéristiques pour gérer les compteurs de référence.
 *
 * Par défaut, on utilise comme implémentation la classe std::shared_ptr.
 */
template <typename InstanceType>
struct RefTraits
{
  static constexpr int TagId = arcaneImplGetRefTagId(static_cast<InstanceType*>(nullptr));
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename InstanceType, int TagType>
struct RefTraitsTagId;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Accesseur des méthodes de gestion de compteurs de référence.
 *
 * Le classe T doit définir deux méthodes addReference() et removeReference()
 * pour gérer les compteurs de références. removeReference() doit détruire
 * l'instance si le compteur arrive à zéro.
 */
template <class T>
class ReferenceCounterAccessor
{
 public:

  static void addReference(T* t)
  {
    if constexpr (requires { t->_internalAddReference(); })
      t->_internalAddReference();
    else
      t->addReference();
  }
  static void removeReference(T* t)
  {
    if constexpr (requires { t->_internalRemoveReference(); }) {
      bool need_destroy = t->_internalRemoveReference();
      if (need_destroy)
        delete t;
    }
    else
      t->removeReference();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro pour déclarer les méthodes virtuelles gérant les compteurs
 * de référence.
 *
 * Cette macro s'utilise de la même manière que les déclarations
 * de méthodes d'une interface. Elle permet de définir des méthodes virtuelles
 * pure pour accèder aux informations des compteurs de référence.
 *
 * La classe implémentant l'interface doit utiliser la macro
 * ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS() pour définir les
 * méthodes virtuelles utilisées.
 *
 * \code
 * class IMyInterface
 * {
 *   ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();
 *  public:
 *   virtual ~IMyInterface() = default;
 *  public:
 *   virtual void myMethod1() = 0;
 * };
 * \endcode
 */
#define ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS() \
 private: \
\
  template <typename T> friend class ::Arccore::ExternalReferenceCounterAccessor; \
  template <typename T> friend class Arcane::ReferenceCounterAccessor; \
\
 public: \
\
  using ReferenceCounterTagType = ::Arcane::ReferenceCounterTag; \
  virtual ::Arcane::ReferenceCounterImpl* _internalReferenceCounter() = 0; \
  virtual void _internalAddReference() = 0; \
  [[nodiscard]] virtual bool _internalRemoveReference() = 0
// NOTE: les classes 'friend' sont nécessaires pour l'accès au destructeur.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro pour déclarer qu'une classe utilise un compteur de
 * référence.
 *
 * La macro doit être utilisée en dehors de tout namespace. Par exemple:
 *
 * \code
 * namespace MyNamespace
 * {
 *   class MyClass;
 * }
 *
 * ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(MyNamespace::MyClass);
 * \endcode
 *
 * Il faudra ensuite utiliser la macro
 * ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS() dans le fichier source pour
 * définir les méthodes et types nécessaires
 */
#define ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(class_name) \
  namespace Arcane \
  { \
    template <> \
    struct RefTraits<class_name> \
    { \
      static constexpr int TagId = ::Arcane::REF_TAG_REFERENCE_COUNTER; \
    }; \
    constexpr inline int arcaneImplGetRefTagId(class_name*) \
    { \
      return ::Arcane::REF_TAG_REFERENCE_COUNTER; \
    } \
    template <> \
    class ReferenceCounterAccessor<class_name> \
    : public ExternalReferenceCounterAccessor<class_name> \
    {}; \
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
using Arcane::ReferenceCounterTag;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
