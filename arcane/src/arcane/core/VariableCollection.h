// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableCollection.h                                        (C) 2000-2025 */
/*                                                                           */
/* Collection de variables.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLECOLLECTION_H
#define ARCANE_CORE_VARIABLECOLLECTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/core/VariableRef.h"
#include "arcane/core/SharedReference.h"
#include "arcane/utils/AutoRef.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableCollection;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT VariableCollectionEnumerator
{
 private:
  friend class VariableCollection;
 public:
  VariableCollectionEnumerator(const VariableCollection& col);
 public:
  bool operator++()
  {
    ++m_index;
    return m_index<m_count;
  }
  IVariable* operator*()
  {
    return m_collection[m_index];
  }
  IVariable* current()
  {
    return this->operator*();
  }
  bool moveNext()
  {
    return this->operator++();
  }
  void reset()
  {
    m_index = -1;
  }
 private:
  Integer m_index;
  Integer m_count;
  ConstArrayView<IVariable*> m_collection;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Collection de variables.
 *
 * Ce type à une sémantique par référence (comme la classe SharedArray).
 */
class ARCANE_CORE_EXPORT VariableCollection
{
 private:

  class Impl
  : public SharedReference
  {
   public:
    const Array<IVariable*>& variables() const { return m_variables; }
    Array<IVariable*>& variables() { return m_variables; }
   public:
    void deleteMe() override { delete this; }
   private:
    UniqueArray<IVariable*> m_variables;
  };

 private:

  typedef Array<IVariable*> BaseClass;

 public:
  
  friend class VariableCollectionEnumerator;
  typedef VariableCollectionEnumerator Enumerator;

 public:

  VariableCollection();
  VariableCollection(const Enumerator& rhs);

 public:

  void add(IVariable* var)
  {
    _values().add(var);
  }

  void add(VariableRef& var)
  {
    _values().add(var.variable());
  }

  IVariable* front()
  {
    return _values()[0];
  }

  Integer count() const
  {
    return _values().size();
  }

  //! Applique le fonctor \a f à tous les éléments de la collection
  template<class Function> Function
  each(Function f)
  {
    std::for_each(_values().begin(),_values().end(),f);
    return f;
  }

  void clear()
  {
    _values().clear();
  }

  bool empty() const
  {
    return _values().empty();
  }

  VariableCollection clone() const
  {
    VariableCollection new_collection;
    new_collection._values().copy(_values());
    return new_collection;
  }

  VariableCollectionEnumerator enumerator() const
  {
    return VariableCollectionEnumerator(*this);
  }

  bool contains(IVariable* v) const
  {
    return _values().contains(v);
  }

  bool contains(VariableRef& v) const
  {
    return _values().contains(v.variable());
  }

  //! Trie la liste par ordre croissant ou décroissant des noms des variables
  void sortByName(bool is_ascendent);

 private:

  const Array<IVariable*>& _values() const { return m_p->variables(); }
  Array<IVariable*>& _values() { return m_p->variables(); }

 private:

  AutoRefT<Impl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

