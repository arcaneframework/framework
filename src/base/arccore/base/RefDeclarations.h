// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RefDeclarations.h                                           (C) 2000-2020 */
/*                                                                           */
/* Déclarations liées à la gestion des références sur une instance.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_REFDECLARATIONS_H
#define ARCCORE_BASE_REFDECLARATIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

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

namespace Arccore
{

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
 *    typedef ReferenceCounterTag ReferenceCounterTagType;
 *   public:
 *    void addReference();
 *    void removeReference();
 * };
 * \endcode
 */
struct ReferenceCounterTag {};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

constexpr int REF_TAG_SHARED_PTR = 0;
constexpr int REF_TAG_REFERENCE_COUNTER = 1;

/*!
 * \brief Caractéristiques pour gérer les compteurs de référence.
 *
 * Par défaut, on utilise comme implémentation la classe std::shared_ptr.
 */
template<typename InstanceType,class T = void>
struct RefTraits
{
  static constexpr int TagId = REF_TAG_SHARED_PTR;
};

/*!
 * \brief Spécialisation de la classe gérant un compteur de référence
 * si la classe utilise le tag 'ReferenceCounterTag'.
 *
 * Dans ce cas, on utilise 'ReferenceCounter' comme implémentation.
 */
template<typename InstanceType>
struct RefTraits<InstanceType,std::enable_if_t<std::is_same_v<typename InstanceType::ReferenceCounterTagType,ReferenceCounterTag>>>
{
  static constexpr int TagId = REF_TAG_REFERENCE_COUNTER;
};

template<typename InstanceType,
         int ImplTagId = RefTraits<InstanceType>::TagId>
class Ref;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename InstanceType,int TagType>
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
template<class T>
class ReferenceCounterAccessor
{
 public:
  static void addReference(T* t) { t->addReference(); }
  static void removeReference(T* t) { t->removeReference(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T>
class ARCCORE_EXPORT ExternalReferenceCounterAccessor
{
 public:
  static void addReference(T* t);
  static void removeReference(T* t);
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
 public:\
  typedef Arccore::ReferenceCounterTag ReferenceCounterTagType;  \
  virtual Arccore::ReferenceCounterImpl* _internalReferenceCounter() =0; \
  virtual void addReference() =0;\
  virtual void removeReference() =0

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro pour déclarer qu'une classe utilise un compteur de
 * référence.
 *
 * La macro doit être utilisée dans le namespace Arccore. Par exemple:
 *
 * \code
 * namespace MyNamespace
 * {
 *   class MyClass;
 * }
 *
 * namespace Arccore
 * {
 *   ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(MyNamespace::MyClass);
 * }
 * \endcode
 *
 * Il faudra ensuite utiliser la macro
 * ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS() dans le fichier source pour
 * définir les méthodes et types nécessaires
 */
#define ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(class_name) \
template<> \
struct RefTraits<class_name> \
{\
  static constexpr int TagId = Arccore::REF_TAG_REFERENCE_COUNTER;\
};\
template<>\
class ReferenceCounterAccessor<class_name>\
: public ExternalReferenceCounterAccessor<class_name>\
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

