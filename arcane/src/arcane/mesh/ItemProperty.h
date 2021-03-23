// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemProperty.h                                             (C) 2000-2015  */
/*                                                                           */
/* Property on item to handle new connectivities and future mesh properties  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DOF_ITEMPROPERTY_H
#define ARCANE_DOF_ITEMPROPERTY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/Array2.h"
#include "arcane/utils/MultiArray2.h"
#include "arcane/ArcaneTypes.h"
#include "arcane/IItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
class ItemScalarProperty
{
 public:

  /** Constructeur de la classe */
  ItemScalarProperty() : m_default_value(DataType()) {}

 public:

  template <class AbstractFamily>
  void resize(AbstractFamily* item_family, const DataType default_value) // SDC Template AbstractFamily necessaire pour applications IFPEN
  {
    m_data.resize(item_family->maxLocalId(),default_value);
    m_default_value = default_value;
  }

  Integer size() const {return m_data.size();}

  template<class AbstractItem>
  DataType& operator[](const AbstractItem item)       {return m_data[item.localId()];}

  template<class AbstractItem>
  DataType  operator[](const AbstractItem item) const {return m_data[item.localId()];}

  void print() const
  {
    std::cout << "== CONNECTIVITY ITEM PROPERTY CONTAINS " << std::endl;
    for (Arcane::Integer i = 0; i < m_data.size();++i)
      {
        std::cout << "\""<< m_data[i] << "\"" << std::endl;
      }
    std::cout << std::endl;
  }

  ConstArrayView<DataType> view() {return m_data.constView();}

  void updateSupport(Int32ConstArrayView new_to_old_ids)
  {
    UniqueArray<DataType> old_data(m_data);
    Integer new_size = new_to_old_ids.size();
    m_data.resize(new_size);
    if (new_size > old_data.size()) old_data.resize(new_size,m_default_value);// padd new items
    for (Integer i = 0; i < new_size; ++i)
      {
        m_data[i] = old_data[new_to_old_ids[i]];
      }
  }

  void copy(const ItemScalarProperty<DataType>& item_property_from)
  {
    m_data.copy(item_property_from.m_data); // copy values
    m_default_value = item_property_from.m_default_value;
  }

 private:

//  UniqueArray<DataType> m_data;
  SharedArray<DataType> m_data; // ItemProperty used in IFPEN applications as a shared object (Arcane-like...).
  DataType m_default_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
class ItemArrayProperty
{
 public:

  /** Constructeur de la classe */
  ItemArrayProperty() {}

 public:

  template <class AbstractFamily>
  void resize(AbstractFamily* item_family, const Integer nb_element_per_item, const DataType default_value) // SDC Template AbstractFamily necessaire pour applications IFPEN
  {
    Integer dim1_old_size = m_data.dim1Size();
    Integer dim2_old_size = m_data.dim2Size();
    Integer dim1_new_size = item_family->maxLocalId();
    Integer dim2_new_size = nb_element_per_item;
    m_data.resize(dim1_new_size,dim2_new_size);
    // Fill new data
    // if new dim 1 larger, enter loop
    for (Integer i = dim1_old_size; i < dim1_new_size; ++i)
      {
        for (Integer j = 0; j < dim2_new_size; ++j) m_data[i][j] = default_value;
      }
    // test if dim2 larger before loop
    if (dim2_new_size > dim2_old_size)
      {
        for (Arcane::Integer i = 0; i < dim1_old_size; ++i)
          {
            for (Integer j = dim2_old_size; j < dim2_new_size;++j) m_data[i][j] = default_value;
          }
      }
  }

  Integer dim1Size() const {return m_data.dim1Size();}
  Integer dim2Size() const {return m_data.dim2Size();}

  template<class AbstractItem>
  ArrayView<DataType> operator[](AbstractItem item) { return m_data[item.localId()]; }

  template<class AbstractItem>
  ConstArrayView<DataType> operator[](AbstractItem item) const { return m_data[item.localId()]; }

  void updateSupport(Int32ConstArrayView new_to_old_ids)
  {
    UniqueArray2<DataType> old_data(m_data);
    Integer new_dim1_size = new_to_old_ids.size();
    Integer dim2_size = m_data.dim2Size();
    m_data.resize(new_dim1_size);
    for (Integer i = 0; i < new_dim1_size; ++i)
      {
        for (Integer j = 0; j < dim2_size;++j) m_data[i][j] = old_data[new_to_old_ids[i]][j];
      }
  }

  void copy(const ItemArrayProperty<DataType>& item_property_from)
  {
    m_data.copy(item_property_from.m_data); // copy values
  }

 private:

  UniqueArray2<DataType> m_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
class ItemMultiArrayProperty
{
 public:

  /** Constructeur de la classe */
  ItemMultiArrayProperty() {}

 public:

  typedef MultiArray2<DataType> InternalArray2;

  // nb_element per item est de taille IItemFamily::maxLocalId
  template <class AbstractFamily>
  void resize(AbstractFamily* item_family, const Arcane::IntegerConstArrayView nb_element_per_item, const DataType default_value) // SDC Template AbstractFamily necessaire pour applications IFPEN
  {
    ARCANE_ASSERT((nb_element_per_item.size() == item_family->maxLocalId()),
                  ("In item property resize : nb_element_per_item must have size IItemFamilyy::maxLocalId"))
    Integer dim1_old_size = m_data.dim1Size();
    Integer dim1_new_size = nb_element_per_item.size();
    IntegerUniqueArray    dim2_old_sizes(m_data.dim2Sizes());
    IntegerConstArrayView dim2_new_sizes(nb_element_per_item);
    m_data.resize(nb_element_per_item);
    // Fill new data with default value
    // Dim1 size larger
    for (Arcane::Integer i = dim1_old_size; i < dim1_new_size; ++i)
      {
        for (Arcane::Integer j = 0; j < dim2_new_sizes[i]; ++j)
          {
            m_data[i][j] = default_value;
          }
      }
    // Dim2 size larger (eventually, cannot know a priori, need to loop).
    for (Arcane::Integer i = 0; i < dim1_old_size; ++i)
      {
        for (Arcane::Integer j = dim2_old_sizes[i]; j < dim2_new_sizes[i]; ++j)
          {
            m_data[i][j] = default_value;
          }
      }
  }

  Integer dim1Size() const {return m_data.dim1Size();}
  IntegerConstArrayView dim2Sizes() const {return m_data.dim2Sizes();}

  template<class AbstractItem>
  ArrayView<DataType>      operator[](const AbstractItem item)       {return m_data[item.localId()];}

  template<class AbstractItem>
  ConstArrayView<DataType> operator[](const AbstractItem item) const {return m_data[item.localId()];}

  void updateSupport(Int32ConstArrayView new_to_old_ids)
  {
    UniqueMultiArray2<DataType> old_data(m_data);
    Integer new_dim1_size = new_to_old_ids.size();
    IntegerConstArrayView dim2_sizes = m_data.dim2Sizes();
    IntegerUniqueArray new_dim2_sizes(new_dim1_size);
    // Compute new sizes
    for (Integer i = 0; i < new_dim1_size; ++i)
      {
        new_dim2_sizes[i] = dim2_sizes[new_to_old_ids[i]];
      }
    m_data.resize(new_dim2_sizes);
    for (Integer i = 0; i < new_dim1_size; ++i)
      {
        for (Integer j = 0; j < new_dim2_sizes[i];++j) m_data[i][j] = old_data[new_to_old_ids[i]][j];
      }
  }

  void copy(const ItemMultiArrayProperty<DataType>& item_property_from)
  {
    m_data = item_property_from.m_data.constView(); // copy values
  }

 private:

//  UniqueMultiArray2<DataType> m_data;
  SharedMultiArray2<DataType> m_data; // ItemProperty used in IFPEN applications as a shared object (Arcane-like...).
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ITEMPROPERTY_H_ */
