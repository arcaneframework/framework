// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GenericRegisterer.h                                         (C) 2000-2022 */
/*                                                                           */
/* Enregistreur générique de types globaux.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_GENERICREGISTERER_H
#define ARCANE_UTILS_GENERICREGISTERER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe template pour gérer une liste globale permettant d'enregistrer
 * des fabriques.
 *
 * Cette classe utilise le Curiously recurring template pattern (CRTP). Le
 * paramètre template doit être la classe dérivée et doit avoir une méthode
 * globalRegistererInfo() comme suit:
 * \code
 * class MyRegisterer
 * : public GenericRegisterer<MyRegisterer>
 * {
 *  public:
 *   GenericRegisterer<MyRegisterer>::Info& registererInfo();
 * };
 * \endcode
 */
template <typename Type>
class ARCANE_UTILS_EXPORT GenericRegisterer
{
 protected:

  class Info
  {
    friend GenericRegisterer<Type>;

   public:

    Type* firstRegisterer() const { return m_first_registerer; }
    Int32 nbRegisterer() const { return m_nb_registerer; }

   private:

    Type* m_first_registerer = nullptr;
    Int32 m_nb_registerer = 0;
  };

 public:

  using InstanceType = Type;

 public:

  GenericRegisterer() noexcept
  {
    _init();
  }

 public:

  //! Instance précédente (nullptr si la première)
  InstanceType* previousRegisterer() const { return m_previous; }

  //! Instance suivante (nullptr si la dernière)
  InstanceType* nextRegisterer() const { return m_next; }

 public:

  //! Accès au premier élément de la chaine d'enregistreur
  static InstanceType* firstRegisterer()
  {
    return Type::registererInfo().firstRegisterer();
  }

  //! Nombre d'enregisteur de service dans la chaine
  static Integer nbRegisterer()
  {
    return Type::registererInfo().nbRegisterer();
  }

 private:

  InstanceType* m_previous = nullptr;
  InstanceType* m_next = nullptr;

 private:

  void _init() noexcept
  {
    Type* current_instance = static_cast<Type*>(this);
    // ATTENTION: Cette méthode est appelée depuis un constructeur global
    // (donc avant le main()) et il ne faut pas faire d'exception dans ce code.
    InstanceType* first = Type::registererInfo().firstRegisterer();
    if (!first) {
      Type::registererInfo().m_first_registerer = current_instance;
      m_previous = nullptr;
      m_next = nullptr;
    }
    else {
      InstanceType* next = first->nextRegisterer();
      m_next = first;
      Type::registererInfo().m_first_registerer = current_instance;
      if (next)
        next->m_previous = current_instance;
    }
    ++Type::registererInfo().m_nb_registerer;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
