// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GenericRegisterer.h                                         (C) 2000-2025 */
/*                                                                           */
/* Enregistreur générique de types globaux.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_GENERICREGISTERER_H
#define ARCCORE_BASE_GENERICREGISTERER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class ARCCORE_BASE_EXPORT GenericRegistererBase
{
 protected:

  [[noreturn]] void doErrorConflict();
  [[noreturn]] void doErrorNonZeroCount();
};

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
class GenericRegisterer
: public GenericRegistererBase
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
    Info& reg_info = Type::registererInfo();
    Type* current_instance = static_cast<Type*>(this);
    // ATTENTION: Cette méthode est appelée depuis un constructeur global
    // (donc avant le main()) et il ne faut pas faire d'exception dans ce code.
    InstanceType* first = reg_info.firstRegisterer();
    if (!first) {
      reg_info.m_first_registerer = current_instance;
      m_previous = nullptr;
      m_next = nullptr;
    }
    else {
      InstanceType* next = first->nextRegisterer();
      m_next = first;
      reg_info.m_first_registerer = current_instance;
      if (next)
        next->m_previous = current_instance;
    }
    ++reg_info.m_nb_registerer;

    // Check integrity
    auto* p = reg_info.firstRegisterer();
    Integer count = reg_info.nbRegisterer();
    while (p && count > 0) {
      p = p->nextRegisterer();
      --count;
    }
    if (p) {
      doErrorConflict();
    }
    else if (count > 0) {
      doErrorNonZeroCount();
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
