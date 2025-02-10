// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Dispatchers.h                                               (C) 2000-2025 */
/*                                                                           */
/* Conteneur des dispatchers.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_DISPATCHERS_H
#define ARCCORE_MESSAGEPASSING_DISPATCHERS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/IDispatchers.h"

#include "arccore/base/BuiltInDataTypeContainer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du conteneur des dispatchers.
 */
class ARCCORE_MESSAGEPASSING_EXPORT Dispatchers
: public IDispatchers
{
 private:

  template <typename DataType>
  class ContainerTraits
  {
   public:

    using InstanceType = ITypeDispatcher<DataType>*;
  };

 public:

  Dispatchers();
  ~Dispatchers() override;

 public:

  ITypeDispatcher<char>* dispatcher(char* v) override { return m_container.instance(v); }
  ITypeDispatcher<signed char>* dispatcher(signed char* v) override { return m_container.instance(v); }
  ITypeDispatcher<unsigned char>* dispatcher(unsigned char* v) override { return m_container.instance(v); }
  ITypeDispatcher<short>* dispatcher(short* v) override { return m_container.instance(v); }
  ITypeDispatcher<unsigned short>* dispatcher(unsigned short* v) override { return m_container.instance(v); }
  ITypeDispatcher<int>* dispatcher(int* v) override { return m_container.instance(v); }
  ITypeDispatcher<unsigned int>* dispatcher(unsigned int* v) override { return m_container.instance(v); }
  ITypeDispatcher<long>* dispatcher(long* v) override { return m_container.instance(v); }
  ITypeDispatcher<unsigned long>* dispatcher(unsigned long* v) override { return m_container.instance(v); }
  ITypeDispatcher<long long>* dispatcher(long long* v) override { return m_container.instance(v); }
  ITypeDispatcher<unsigned long long>* dispatcher(unsigned long long* v) override { return m_container.instance(v); }
  ITypeDispatcher<float>* dispatcher(float* v) override { return m_container.instance(v); }
  ITypeDispatcher<double>* dispatcher(double* v) override { return m_container.instance(v); }
  ITypeDispatcher<long double>* dispatcher(long double* v) override { return m_container.instance(v); }
  ITypeDispatcher<BFloat16>* dispatcher(BFloat16* v) override { return m_container.instance(v); }
  ITypeDispatcher<Float16>* dispatcher(Float16* v) override { return m_container.instance(v); }

  IControlDispatcher* controlDispatcher() override { return m_control; }
  ISerializeDispatcher* serializeDispatcher() override { return m_serialize; }

 public:

  template <typename DataType> void setDispatcher(ITypeDispatcher<DataType>* x)
  {
    DataType* ptr = nullptr;
    m_container.instance(ptr) = x;
  }

  void setDispatcher(IControlDispatcher* x) { m_control = x; }
  void setDispatcher(ISerializeDispatcher* x) { m_serialize = x; }

  //! Indique si lors de la destruction on appelle l'opérateur delete sur les instances (faux par défaut)
  void setDeleteDispatchers(bool v) { m_is_delete_dispatchers = v; }

 private:

  Arccore::BuiltInDataTypeContainer<ContainerTraits> m_container;

  IControlDispatcher* m_control = nullptr;
  ISerializeDispatcher* m_serialize = nullptr;

  bool m_is_delete_dispatchers = false;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
