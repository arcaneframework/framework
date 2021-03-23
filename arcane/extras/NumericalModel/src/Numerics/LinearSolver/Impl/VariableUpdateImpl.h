// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef VARIABLEUPDATE_H
#define VARIABLEUPDATE_H

#include <arcane/ArcaneVersion.h>
#include <arcane/ItemGroup.h>
#include <arcane/IVariable.h>
#include <arcane/datatype/DataTypes.h>
#if(ARCANE_VERSION<10600)
#endif
#if(ARCANE_VERSION>=10600) && (ARCANE_VERSION<10800)
#include <arcane/utils/Array2.h>
#endif
#if(ARCANE_VERSION>=10800)
#include <arcane/utils/Array2.h>
#include <arcane/utils/Array2View.h>
#include <arcane/utils/UtilsTypes.h>
#endif

#include <arcane/IVariableAccessor.h>

#include "Numerics/LinearSolver/IIndexManager.h"

/* \todo 
 * A terme il faudra implementer l'ensemble des formes d'accesseurs suivant:
 * ArrayView<Real>       asArrayReal ()=0 // le seul cas implémenté, vérifié par assert
 * ArrayView<Real2> 	asArrayReal2 ()=0
 * ArrayView<Real3> 	asArrayReal3 ()=0
 * ArrayView<Real2x2> 	asArrayReal2x2 ()=0
 * ArrayView<Real3x3> 	asArrayReal3x3 ()=0
 * Array2View<Real> 	asArray2Real ()=0
 * Array2View<Real2> 	asArray2Real2 ()=0
 * Array2View<Real3> 	asArray2Real3 ()=0
 * Array2View<Real3x3> 	asArray2Real3x3 ()=0
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Initializer par mise à zero
class MeshVariableZeroInitializer : 
public IIndexManager::Initializer
{
 protected:
  IVariable * m_ivar;

 public:
  MeshVariableZeroInitializer(IVariable * ivar)
    : m_ivar(ivar)
    {
      ARCANE_ASSERT((DT_Real == m_ivar->dataType()),("Incompatible data type"));
      ARCANE_ASSERT((1 == m_ivar->dimension()),("Incompatible dimension"));
    }

  virtual ~MeshVariableZeroInitializer()
    {
      ;
    }

  void init(const ConstArrayView<Item> & items, Array<Real> & values)
    {      
      ARCANE_ASSERT((items.size() == values.size()),("Incompatible items and values sizes"));

      for(Integer i=0;i<items.size();++i)
        {
          ARCANE_ASSERT((items[i].kind() == m_ivar->itemKind()),("Bad item kind"));
          values[i] = 0.;
        }
    }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Initializer par copie
class MeshVariableCopyInitializer : 
public IIndexManager::Initializer
{
 protected:
  IVariable * m_ivar;
  RealArrayView m_data;

 public:
  MeshVariableCopyInitializer(IVariable * ivar)
    : m_ivar(ivar),
    m_data(ivar->accessor()->asArrayReal())
    {
      ARCANE_ASSERT((DT_Real == m_ivar->dataType()),("Incompatible data type"));
      ARCANE_ASSERT((1 == m_ivar->dimension()),("Incompatible dimension"));
    }

  virtual ~MeshVariableCopyInitializer()
    {
      ;
    }

  void init(const ConstArrayView<Item> & items, Array<Real> & values)
    {
      ARCANE_ASSERT((items.size() == values.size()),("Incompatible items and values sizes"));

      // Weak integrity check
      // We are assuming that, for partial variables, items is the item group
      // on which the variable is built
#warning "TODO: Temporary patch to handle partial variables"
      if(m_ivar->isPartial())
        ARCANE_ASSERT((items.size() == m_ivar->itemGroup().size()),("Incompatible items and group sizes on partial variable"));

      for(Integer i=0;i<items.size();++i)
        {
          const Item & item = items[i];
          ARCANE_ASSERT((item.kind() == m_ivar->itemKind()),("Bad item kind"));
          values[i] = m_data[m_ivar->isPartial() ? i : item.localId()];
        }
    }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Updater par copie
class MeshVariableCopyUpdater : 
public IIndexManager::Updater
{
 protected:
  IVariable * m_ivar;
  RealArrayView m_data;

 public:
  MeshVariableCopyUpdater(IVariable * ivar)
    : m_ivar(ivar),
    m_data(ivar->accessor()->asArrayReal())
    {
      ARCANE_ASSERT((DT_Real == m_ivar->dataType()),("Incompatible data type"));
      ARCANE_ASSERT((1 == m_ivar->dimension()),("Incompatible dimension"));
    }

  virtual ~MeshVariableCopyUpdater()
    {
      ;
    }

  void update(const ConstArrayView<Item> & items, const Array<Real> & values)
    {
      ARCANE_ASSERT((items.size() == values.size()),("Incompatible items and values sizes"));
    
      // Weak integrity check
      // We are assuming that, for partial variables, items is the item group
      // on which the variable is built
#warning "TODO: Temporary patch to handle partial variables"
      if(m_ivar->isPartial())
        ARCANE_ASSERT((items.size() == m_ivar->itemGroup().size()),("Incompatible items and group sizes on partial variable"));

      for(Integer i=0;i<items.size();++i)
        {
          const Item & item = items[i];
          ARCANE_ASSERT((item.kind() == m_ivar->itemKind()),("Bad item kind"));
          m_data[m_ivar->isPartial() ? i : item.localId()] = values[i];
        }
    }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Updater par copie
class MeshArrayVariablePartialCopyUpdater : 
  public IIndexManager::Updater
{
protected:
  IVariable * m_ivar;
  Array2View<Real> m_data;
  Integer m_offset;
  
public:
 MeshArrayVariablePartialCopyUpdater(IVariable * ivar, Integer offset)
   : m_ivar(ivar),
     m_data(ivar->accessor()->asArray2Real()),
     m_offset(offset)
   
  {
    ARCANE_ASSERT((DT_Real == m_ivar->dataType()),("Incompatible data type"));
    ARCANE_ASSERT((2 == m_ivar->dimension()),("Incompatible dimension"));
    ARCANE_ASSERT((offset>=0),("Negative offset [%d] not allowed",offset));
  }
  
  virtual ~MeshArrayVariablePartialCopyUpdater()
  {
    ;
  }
  
  void update(const ConstArrayView<Item> & items, const Array<Real> & values)
  {
    ARCANE_ASSERT((items.size() == values.size()),("Incompatible items and values sizes"));
    
    // Weak integrity check
    // We are assuming that, for partial variables, items is the item group
    // on which the variable is built
#warning "TODO: Temporary patch to handle partial variables"
    if(m_ivar->isPartial())
      ARCANE_ASSERT((items.size() == m_ivar->itemGroup().size()),("Incompatible items and group sizes on partial variable"));
    
    for(Integer i=0;i<items.size();++i)
    {
      const Item & item = items[i];
      ARCANE_ASSERT((item.kind() == m_ivar->itemKind()),("Bad item kind"));
      m_data[m_ivar->isPartial() ? i : item.localId()][m_offset] = values[i];
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Update par accumulation (ie +=)
class MeshVariableCumulativeUpdater : 
public IIndexManager::Updater
{
 protected:
  IVariable * m_ivar;
  RealArrayView m_data;

 public:
  MeshVariableCumulativeUpdater(IVariable * ivar)
    : m_ivar(ivar),
    m_data(ivar->accessor()->asArrayReal())
    {
      ARCANE_ASSERT((DT_Real == m_ivar->dataType()),("Incompatible data type"));
      ARCANE_ASSERT((1 == m_ivar->dimension()),("Incompatible dimension"));
    }

  virtual ~MeshVariableCumulativeUpdater()
    {
      ;
    }

  void update(const ConstArrayView<Item> & items, const Array<Real> & values)
    {
      ARCANE_ASSERT((items.size() == values.size()),("Incompatible items and values sizes"));

      // Weak integrity check
      // We are assuming that, for partial variables, items is the item group
      // on which the variable is built
#warning "TODO: Temporary patch to handle partial variables"
      if(m_ivar->isPartial())
	  ARCANE_ASSERT((items.size() == m_ivar->itemGroup().size()),("Incompatible items and group sizes on partial variable"));
    
      for(Integer i=0;i<items.size();++i)
        {
          const Item & item = items[i];
          ARCANE_ASSERT((item.kind() == m_ivar->itemKind()),("Bad item kind"));
	  m_data[m_ivar->isPartial() ? i : item.localId()] += values[i];
        }
    }
};


/*---------------------------------------------------------------------------*/

//! Updater par composition de 2 updaters
class MeshVariableDualUpdater : 
public IIndexManager::Updater
{
 protected:
  AutoRefT<IIndexManager::Updater> m_first;
  AutoRefT<IIndexManager::Updater> m_second;

 public:
  MeshVariableDualUpdater(IIndexManager::Updater * first, 
                          IIndexManager::Updater * second)
    : m_first(first), m_second(second)
    {
      ;
    }

  virtual ~MeshVariableDualUpdater()
    {
      ;
    }

  void update(const ConstArrayView<Item> & items, const Array<Real> & values)
    {
      m_first->update(items,values);
      m_second->update(items,values);
    }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* VARIABLEUPDATE_H */
