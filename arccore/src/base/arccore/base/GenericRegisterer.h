// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GenericRegisterer.h                                         (C) 2000-2025 */
/*                                                                           */
/* Generic registerer for global types.                                      */
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
 * \brief Template class to manage a global list for registering
 * factories.
 *
 * This class uses the Curiously Recurring Template Pattern (CRTP). The
 * template parameter must be the derived class and must have a method
 * globalRegistererInfo() as follows:
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

  //! Previous instance (nullptr if it is the first)
  InstanceType* previousRegisterer() const { return m_previous; }

  //! Next instance (nullptr if it is the last)
  InstanceType* nextRegisterer() const { return m_next; }

 public:

  //! Access to the first element of the registerer chain
  static InstanceType* firstRegisterer()
  {
    return Type::registererInfo().firstRegisterer();
  }

  //! Number of service registerers in the chain
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
    // ATTENTION: This method is called from a global constructor
    // (i.e., before main()) and no exceptions should be thrown in this code.
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
