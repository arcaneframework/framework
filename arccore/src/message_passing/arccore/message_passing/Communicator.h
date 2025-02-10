// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Communicator.h                                              (C) 2000-2025 */
/*                                                                           */
/* Communicateur pour l'échange de message.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_COMMUNICATOR_H
#define ARCCORE_MESSAGEPASSING_COMMUNICATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/base/Ref.h"

#include <cstddef>
#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Communicateur pour l'échange de message.
 *
 * Cette classe est une abstraction du communicateur qu'on trouve
 * dans la norme MPI sous le type 'MPI_Comm'.
 *
 * Cette classe permet de conserver de manière générique un communicateur sans
 * connaitre son type exact (par exemple MPI_Commu avec la norme MPI). On utilise
 * pour cela une union.
 *
 * Avant d'utiliser une instance de cette classe, il faut positionner le
 * communicateur nul en appelant la méthode statique setNullCommunicator()
 * avec la valeur du communicateur nul pour l'implémentation utilisée.
 */
class ARCCORE_MESSAGEPASSING_EXPORT Communicator
{
  union _Communicator
  {
    int i;
    long l;
    std::size_t st;
    void* v;
    const void* cv;
  };

  enum Type
  {
    T_Int,
    T_Long,
    T_SizeT,
    T_Ptr,
    T_Null
  };

 public:

  Communicator()
  {
    m_type = T_Null;
    m_communicator = null_communicator;
  }

 public:

  explicit Communicator(void* acommunicator)
  {
    m_type = T_Ptr;
    m_communicator.v = acommunicator;
  }

  explicit Communicator(const void* acommunicator)
  {
    m_type = T_Ptr;
    m_communicator.cv = acommunicator;
  }

  explicit Communicator(int acommunicator)
  {
    m_type = T_Int;
    m_communicator.i = acommunicator;
  }

  explicit Communicator(long acommunicator)
  {
    m_type = T_Long;
    m_communicator.l = acommunicator;
  }

  explicit Communicator(std::size_t acommunicator)
  {
    m_type = T_SizeT;
    m_communicator.st = acommunicator;
  }


 public:

  template<typename T>
  operator const T*() const { return (const T*)m_communicator.cv; }
  template<typename T>
  operator T*() const { return (T*)m_communicator.v; }
  operator int() const { return m_communicator.i; }
  operator long() const { return m_communicator.l; }
  operator size_t() const { return m_communicator.st; }
  void* communicatorAddress() { return &m_communicator; }

 public:

  /*!
   * \brief Indique si le communicateur est valide.
   *
   * Un communicateur est valide s'il est différent du communicateur nul.
   */
  bool isValid() const
  {
    if (m_type==T_Null)
      return false;
    // Si le type de la requête est différent du type
    // de la requête nulle, alors la requête est considérée
    // comme valide.
    if (m_type!=null_communicator_type)
      return true;
    if (m_type==T_Int)
      return m_communicator.i!=null_communicator.i;
    if (m_type==T_Long)
      return m_communicator.l!=null_communicator.l;
    if (m_type==T_SizeT)
      return m_communicator.st!=null_communicator.st;
    if (m_type==T_Ptr)
      return m_communicator.cv!=null_communicator.cv;
    return false;
  }

  static void setNullCommunicator(Communicator r)
  {
    null_communicator = r.m_communicator;
    null_communicator_type = r.m_type;
  }

  void reset()
  {
    m_communicator = null_communicator;
    m_type = T_Null;
  }

  void print(std::ostream& o) const;

  friend inline std::ostream& operator<<(std::ostream& o,const Communicator& pcommunicator)
  {
    pcommunicator.print(o);
    return o;
  }

  //! \internal
  Int32 _type() const { return m_type; }

 private:

  int m_type = T_Null;
  _Communicator m_communicator;
  static _Communicator null_communicator;
  static int null_communicator_type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

